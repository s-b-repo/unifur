//! Geometry-kernel selection: axioms that vary by geometry (roadmap Phase 33,
//! audit fix 13).
//!
//! A reasoner trained only on Euclidean figures over-applies Euclidean
//! assumptions -- the sharpest being "parallel lines never meet", which is
//! false projectively, false spherically, and true only for one of the
//! hyperbolic parallels. The fix is to stop treating the axioms as background
//! knowledge and start treating them as a *declared environment*: a geometry
//! kind, the axioms it validates, and the theorems the engine is therefore
//! allowed to fire.
//!
//! This module carries that declaration plus exact incidence models over [`Q`]
//! for each geometry -- the homogeneous projective plane (with cross ratio),
//! the affine plane (parallel lines meeting at infinity, ratios along a line,
//! no lengths), the Cayley-Klein disk for hyperbolic incidence (through a
//! point off a line pass many non-meeting lines), and the ray/great-circle
//! model of spherical incidence (any two lines meet). Incidence and cross
//! ratios are exact; angle *sums* are not rational quantities, so they are
//! reported as numeric evidence and never as facts -- the same rule the kernel
//! applies to a diagram that merely looks parallel.
//!
//! # What lives here
//!
//! - [`GeometryKind`]: the declared environment, with the [`Axiom`]s it
//!   validates (`axioms`) and the theorems the engine may therefore fire
//!   (`permits`). A rule the environment does not permit is refused by name,
//!   with the permitted list attached, rather than quietly not firing.
//! - [`GeometryKind::check_axioms`]: a scene read under a declared geometry,
//!   checked against that geometry's axioms, reporting contradictions
//!   ([`AxiomViolation`]) separately from statements the axioms leave
//!   undetermined ([`Underdetermination`]) -- a "parallel" pair in the
//!   hyperbolic plane is not false, it is underspecified, and a report that
//!   cannot tell those apart teaches the wrong lesson.
//! - [`ProjectivePlane`]: homogeneous coordinates over [`Q`], exact line
//!   intersection, and the exact cross ratio of four collinear points -- the
//!   projective invariant that takes the place of length and angle.
//! - [`AffinePlane`]: parallel lines meeting at a point at infinity, exact
//!   ratios along a line, and a *named* refusal for the lengths it lacks.
//! - [`CayleyKleinDisk`]: exact hyperbolic incidence against the absolute
//!   conic, where the many-parallels property is exhibited rather than
//!   asserted: exact lines through one point off a line, none of which meets
//!   that line.
//! - [`SphericalModel`]: ray and great-circle incidence, where any two lines
//!   meet, with the meeting direction exact.
//! - [`Incidence`] against [`NumericEvidence`]: the honesty boundary as a
//!   type-level distinction. Incidence is a decision; an angle sum is a
//!   number, and [`NumericEvidence::into_fact`] refuses to promote it,
//!   because a sum of three `acos`es that prints as `180.0` is not `= 180`.
//!
//! # What is deliberately absent
//!
//! Hyperbolic trigonometry. `sinh` and `cosh` of a hyperbolic distance are
//! not in the kernel's quadratic field, so an exact hyperbolic angle needs a
//! different number type; approximating one would put a float exactly where
//! the kernel's discipline is that there is not one. So [`hyperbolic_trig`]
//! refuses, and the hyperbolic claims this module proves are the incidence
//! ones. Also absent: a hyperbolic *diagram* (the model carries exact
//! coordinates, not a drawing), and any proof that the three non-Euclidean
//! axiom tables are mutually consistent.

use crate::geomkernel::{
    Angle3, Constraint, Fact, Frac, GeometryError, KPoint, Q, QSqrt, SceneGraph, Segment,
};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

// --------------------------------------------------------- the refusals ---

/// What the environment refuses, and why.
///
/// The kernel's [`GeometryError`] covers non-degeneracy, and this module
/// reuses it rather than growing a second hierarchy: a zero homogeneous triple
/// is [`GeometryError::EmptyGeometry`], two coincident great circles are
/// [`GeometryError::CoincidentLines`]. These variants are the failures the
/// *environment* has and the kernel has no vocabulary for -- a geometry nobody
/// declared, a rule the declared geometry does not license, a quantity this
/// module will not approximate, and numeric evidence offered as a fact.
#[derive(Debug, Clone, PartialEq)]
pub enum NonEuclideanError {
    /// A `SceneGraph::geometry` string naming no geometry in
    /// [`GeometryKind::ALL`].
    UnknownGeometry {
        name: String,
    },
    /// An environment applied to a scene that declares a different one. The
    /// two cannot be reconciled silently: reading a spherical figure under
    /// the Euclidean rules is the over-application this module stops.
    GeometryMismatch {
        declared: GeometryKind,
        scene: String,
    },
    /// A rule or theorem the declared geometry does not license, with the
    /// ones it does -- so the caller learns what to fire instead, rather than
    /// only what went wrong.
    TheoremNotPermitted {
        geometry: GeometryKind,
        theorem: String,
        permitted: Vec<String>,
    },
    /// A quantity needing a number type this module does not have.
    /// Approximating it would be the failure the whole exercise is about, so
    /// the answer is a refusal with the reason attached.
    OutOfScope {
        detail: String,
    },
    /// Numeric evidence asked to become a fact. Always an error: a sum of
    /// three angles is not a rational, and `180.0000001` is not `180`.
    EvidenceIsNotAFact {
        quantity: String,
    },
    /// A cross ratio of four points that are not four distinct points on one
    /// line. The cross ratio is a decision about exactly four distinct
    /// collinear points; anything else has no value rather than a surprising
    /// one.
    DegenerateCrossRatio {
        detail: String,
    },
}

impl fmt::Display for NonEuclideanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownGeometry { name } => write!(
                f,
                "no geometry named '{name}'; the declared environments are {}",
                GeometryKind::names().join(", ")
            ),
            Self::GeometryMismatch { declared, scene } => write!(
                f,
                "this environment is {declared} but the scene declares '{scene}': reading a \
                 figure under axioms it did not declare is the error this module exists to stop"
            ),
            Self::TheoremNotPermitted { geometry, theorem, permitted } => write!(
                f,
                "{geometry} does not license '{theorem}'; it licenses: {}",
                permitted.join(", ")
            ),
            Self::OutOfScope { detail } => write!(f, "out of scope: {detail}"),
            Self::EvidenceIsNotAFact { quantity } => write!(
                f,
                "{quantity} is numeric evidence, not a fact: it may be reported, quoted and \
                 compared, but it cannot enter the ledger as something established"
            ),
            Self::DegenerateCrossRatio { detail } => {
                write!(f, "cross ratio undefined: {detail}")
            }
        }
    }
}

impl std::error::Error for NonEuclideanError {}

// ------------------------------------------------------ the environment ---

/// A declared geometry: the axioms it validates and the theorems it licenses.
///
/// The kind *is* the environment. There is no global geometry and no default
/// beyond what a scene says: a scene that says nothing is `euclidean` (the
/// kernel's own default), and a scene that says something is read under it,
/// with [`GeometryKind::check_axioms`] holding it to that declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GeometryKind {
    /// Euclid: one parallel through a point, angles of a triangle summing to
    /// 180, and lengths.
    Euclidean,
    /// The affine plane: incidence, parallelism and ratios along a line, with
    /// no metric at all -- length and angle measure are not part of it.
    Affine,
    /// The real projective plane: every two lines meet, and the cross ratio is
    /// the invariant that survives. No order along a line, hence no
    /// betweenness and no "between".
    Projective,
    /// The sphere with great circles: every two lines meet, angles around a
    /// point sum to 360, and a line is a closed circle, so it carries a cyclic
    /// order and no linear one.
    Spherical,
    /// The hyperbolic plane: through a point off a line pass infinitely many
    /// lines that do not meet it, triangle angles sum to less than 180, and a
    /// line has two ideal ends.
    Hyperbolic,
}

impl GeometryKind {
    /// Every declared geometry, in the order a report would list them.
    pub const ALL: [GeometryKind; 5] = [
        GeometryKind::Euclidean,
        GeometryKind::Affine,
        GeometryKind::Projective,
        GeometryKind::Spherical,
        GeometryKind::Hyperbolic,
    ];

    /// The wire name, matching what a `SceneGraph::geometry` string carries.
    pub fn as_str(&self) -> &'static str {
        match self {
            GeometryKind::Euclidean => "euclidean",
            GeometryKind::Affine => "affine",
            GeometryKind::Projective => "projective",
            GeometryKind::Spherical => "spherical",
            GeometryKind::Hyperbolic => "hyperbolic",
        }
    }

    /// Every declared geometry, as a slice.
    pub fn all() -> &'static [GeometryKind] {
        &GeometryKind::ALL
    }

    /// The names a `SceneGraph::geometry` string may take, for an error that
    /// has to tell the reader what would have worked.
    pub fn names() -> Vec<&'static str> {
        GeometryKind::ALL.iter().map(|kind| kind.as_str()).collect()
    }

    /// The geometry a scene declares, parsed from its `geometry` string. An
    /// unparseable name is an error carrying the list of valid ones, because
    /// "elliptical geometry" is a real thing someone will type and reading it
    /// as Euclidean is the bug this module prevents.
    pub fn of_scene(graph: &SceneGraph) -> anyhow::Result<GeometryKind> {
        graph.geometry.parse()
    }

    /// Refuse to read a scene under a geometry other than the one it declares.
    ///
    /// The refusal is returned as the typed error rather than formatted into
    /// an `ensure!`, so a caller that matches on
    /// [`NonEuclideanError::GeometryMismatch`] can recover the declared
    /// environment from the failure -- which is the one thing it wants to know.
    pub fn check_scene(&self, graph: &SceneGraph) -> anyhow::Result<()> {
        let declared = GeometryKind::of_scene(graph)?;
        if declared != *self {
            return Err(NonEuclideanError::GeometryMismatch {
                declared: *self,
                scene: graph.geometry.clone(),
            }
            .into());
        }
        Ok(())
    }

    /// Whether this environment may be the one a scene is read under.
    pub fn matches_scene(&self, graph: &SceneGraph) -> bool {
        GeometryKind::of_scene(graph).is_ok_and(|declared| declared == *self)
    }

    /// The angle-sum axiom of this geometry, or `None` where the geometry has
    /// no angle measure at all. A `None` is not a gap to be filled by
    /// defaulting to 180.
    pub fn angle_sum_axiom(&self) -> Option<Axiom> {
        match self {
            GeometryKind::Euclidean => Some(Axiom::AnglesSumTo180),
            GeometryKind::Spherical => Some(Axiom::AnglesSumTo360),
            GeometryKind::Hyperbolic => Some(Axiom::AnglesSumBelow180),
            GeometryKind::Affine | GeometryKind::Projective => None,
        }
    }

    /// Whether a triangle angle sum of `degrees` is *consistent with* the
    /// angle-sum axiom this geometry declares. This compares numbers, it does
    /// not prove: the sum of three angles is not a rational, and this is the
    /// only place in the module where a float meets an axiom. It returns
    /// `Option` because two of the five geometries declare no such axiom, and
    /// an honest `false` there would read as a refutation.
    pub fn agrees_with_angle_sum(&self, degrees: f64) -> Option<bool> {
        match self.angle_sum_axiom()? {
            Axiom::AnglesSumTo180 => {
                Some((degrees - 180.0).abs() <= ANGLE_SUM_TOLERANCE_DEGREES)
            }
            Axiom::AnglesSumTo360 => Some(degrees > 180.0),
            Axiom::AnglesSumBelow180 => Some(degrees < 180.0),
            // Unreachable: the arm is over the axioms this module knows an
            // angle sum for. Refused rather than defaulted, in keeping with
            // the rest of the module.
            _ => None,
        }
    }
}

impl fmt::Display for GeometryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} geometry", self.as_str())
    }
}

impl FromStr for GeometryKind {
    type Err = anyhow::Error;

    /// Parse a declared geometry. Case and surrounding space are forgiven --
    /// the string arrives from a human-written problem file -- and the
    /// classical synonyms are accepted (`elliptic` for the sphere,
    /// `boltzmann` for the hyperbolic plane), because those are the names the
    /// literature uses. Everything else is an error: leniency toward the
    /// listed synonyms costs nothing, leniency toward unknown names is exactly
    /// the silent defaulting this module refuses.
    fn from_str(text: &str) -> anyhow::Result<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "euclidean" | "euclid" | "euclidian" => Ok(GeometryKind::Euclidean),
            "affine" => Ok(GeometryKind::Affine),
            "projective" => Ok(GeometryKind::Projective),
            "spherical" | "spheric" | "elliptic" | "elliptical" => Ok(GeometryKind::Spherical),
            "hyperbolic" | "boltzmann" | "klein" | "cayley-klein" => Ok(GeometryKind::Hyperbolic),
            _ => Err(NonEuclideanError::UnknownGeometry { name: text.to_string() }.into()),
        }
    }
}

/// The gate a rule engine calls before firing anything: the scene's declared
/// geometry has to license the rule, and the geometry the caller holds has to
/// be the one the scene declares. Returns the environment, so a caller that
/// wanted only permission also learns under which axioms it holds.
pub fn authorize(scene: &SceneGraph, rule_or_theorem: &str) -> anyhow::Result<GeometryKind> {
    let kind = GeometryKind::of_scene(scene)?;
    kind.require_theorem(rule_or_theorem)?;
    Ok(kind)
}

// ------------------------------------------------------------- axioms ---

/// A named axiom, and the geometries that hold it.
///
/// An axiom is a statement *about the environment*, not a fact about a
/// figure. That distinction is the whole mechanism: `AnglesSumTo180` is an
/// axiom of the Euclidean plane and neither an axiom nor a theorem of the
/// hyperbolic one, where it is false, so a rule resting on it may not fire
/// there. Splitting the Euclidean parallel axiom from the others
/// (`ParallelLinesNeverMeet`, `AllLinesMeet`, `PointsAtInfinityExist`) is
/// deliberate: "parallel lines never meet" is really three claims about three
/// different planes, and a reasoner holding only the first will over-apply it
/// in the other two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "axiom", rename_all = "snake_case")]
pub enum Axiom {
    /// Euclid's fifth postulate: through a point off a line, exactly one line
    /// misses it. True in the Euclidean and affine planes; false elsewhere,
    /// and in the hyperbolic plane its failure is the *interesting* property
    /// rather than a defect.
    UniqueParallelThroughPoint,
    /// The hyperbolic property: through a point off a line, infinitely many
    /// lines miss it. The axiom separating hyperbolic from Euclidean, and the
    /// reason a hyperbolic `Parallel` fact must say *which* non-meeting
    /// relation it means rather than only that the lines do not meet.
    InfinitelyManyParallelsThroughPoint,
    /// The negation of the parallel postulate in the other direction: the
    /// sphere, where no two distinct great circles miss each other, and the
    /// hyperbolic plane, where infinitely many do.
    NoUniqueParallel,
    /// In the Euclidean and affine planes, two distinct parallel lines share no
    /// point of the plane. False projectively (they meet at a point at
    /// infinity) and false spherically (great circles always meet).
    ParallelLinesNeverMeet,
    /// In the projective plane and on the sphere, any two distinct lines meet
    /// at exactly one point. The projective version needs no point at
    /// infinity; the spherical one is about great circles, which are the lines
    /// of that plane.
    AllLinesMeet,
    /// The plane has a distinguished line at infinity carrying one ideal point
    /// per direction. True affinely (where the points are the bookkeeping
    /// device for parallelism), projectively (where they are points like any
    /// other) and hyperbolically (as the two ideal ends of each line).
    PointsAtInfinityExist,
    /// The plane contains no point at infinity: a parallel pair simply does
    /// not meet, and nothing marks where it would have. The Euclidean and the
    /// spherical plane, and the affine plane read as itself rather than as
    /// the projective plane minus a line.
    NoPointsAtInfinity,
    /// The line at infinity is not one of the lines of the plane, and every
    /// point of it is reached by a line of the plane. The affine axiom, and
    /// the reason "parallel" in the affine plane is a statement about a point
    /// that is not in the plane.
    LineAtInfinityExcluded,
    /// Cross ratio of four collinear points is a well-defined rational and is
    /// preserved by every collineation. The projective invariant that replaces
    /// both length and angle: it is what a reasoner in a non-metric geometry
    /// can still compute exactly.
    CrossRatioInvariant,
    /// The line carries a linear order, so of three collinear points one is
    /// between the other two. True Euclideanly, affinely and hyperbolically;
    /// false projectively (a projective line is a circle, so betweenness is
    /// cyclic) and spherically (a great circle is closed).
    BetweennessOrdered,
    /// Angles carry a measure, so an angle equality is a statement of the
    /// geometry. True of the three metric geometries, false of the affine
    /// plane (whose transformations do not preserve angles) and of the
    /// projective plane (which has no angles at all).
    AngleMeasureExists,
    /// The Euclidean angle sum: a triangle's angles sum to 180. Named as an
    /// axiom rather than derived, because in a non-Euclidean plane it is the
    /// claim that fails -- the sum is more than 180 on the sphere and less
    /// than 180 in the hyperbolic plane.
    ///
    /// The rename is explicit because `snake_case` would render this
    /// `angles_sum_to180`, and a wire name that runs a digit into a word is a
    /// name a file author will mistype. The same applies to the two axioms
    /// below, for the same reason.
    #[serde(rename = "angles_sum_to_180")]
    AnglesSumTo180,
    /// The spherical angle sum: a spherical triangle's angles sum to *more*
    /// than 180, by an amount equal to its area in units where the whole
    /// sphere is 4*pi, and the angles around a point sum to 360.
    #[serde(rename = "angles_sum_to_360")]
    AnglesSumTo360,
    /// The hyperbolic angle sum: a hyperbolic triangle's angles sum to *less*
    /// than 180, by an amount equal to its hyperbolic area.
    #[serde(rename = "angles_sum_below_180")]
    AnglesSumBelow180,
    /// Desargues' incidence axiom: two triangles perspective from a point are
    /// perspective from a line. Holds in the real projective plane and in
    /// every subgeometry of it, which is how this module certifies it for the
    /// Euclidean, affine and hyperbolic planes at once without a proof per
    /// geometry. Not claimed for the sphere: a great circle is a proper conic
    /// in projective 3-space, not a projective line, and the reduction that
    /// would justify the claim is not carried out here.
    DesarguesIncidence,
    /// A line is unbounded in both directions: a ray from any point of it
    /// never returns to that point. False on the sphere, where a great circle
    /// is a closed curve of finite length, and false in the bare affine plane,
    /// whose lines end at the line at infinity.
    LinesUnboundedBothWays,
    /// Every two distinct points determine exactly one line. The one axiom the
    /// incidence geometry of a plane shares *everywhere*, and the one that is
    /// not enough: it is what lets a reasoner think "the plane" is unambiguous
    /// when this module carries four.
    IncidenceAxiom,
    /// Distances are defined and finite between any two points of the plane.
    /// The metric axiom: true Euclideanly, spherically and hyperbolically
    /// (with three different metrics), false in the affine and projective
    /// planes, where no distance survives the transformation group.
    MetricDefined,
}

impl Axiom {
    /// Every axiom this module knows, in table order.
    pub const ALL: [Axiom; 18] = [
        Axiom::UniqueParallelThroughPoint,
        Axiom::InfinitelyManyParallelsThroughPoint,
        Axiom::NoUniqueParallel,
        Axiom::ParallelLinesNeverMeet,
        Axiom::AllLinesMeet,
        Axiom::PointsAtInfinityExist,
        Axiom::NoPointsAtInfinity,
        Axiom::LineAtInfinityExcluded,
        Axiom::CrossRatioInvariant,
        Axiom::BetweennessOrdered,
        Axiom::AngleMeasureExists,
        Axiom::AnglesSumTo180,
        Axiom::AnglesSumTo360,
        Axiom::AnglesSumBelow180,
        Axiom::DesarguesIncidence,
        Axiom::LinesUnboundedBothWays,
        Axiom::IncidenceAxiom,
        Axiom::MetricDefined,
    ];

    /// The wire name, matching the serde tag's snake_case rendering.
    pub fn as_str(&self) -> &'static str {
        match self {
            Axiom::UniqueParallelThroughPoint => "unique_parallel_through_point",
            Axiom::InfinitelyManyParallelsThroughPoint => "infinitely_many_parallels_through_point",
            Axiom::NoUniqueParallel => "no_unique_parallel",
            Axiom::ParallelLinesNeverMeet => "parallel_lines_never_meet",
            Axiom::AllLinesMeet => "all_lines_meet",
            Axiom::PointsAtInfinityExist => "points_at_infinity_exist",
            Axiom::NoPointsAtInfinity => "no_points_at_infinity",
            Axiom::LineAtInfinityExcluded => "line_at_infinity_excluded",
            Axiom::CrossRatioInvariant => "cross_ratio_invariant",
            Axiom::BetweennessOrdered => "betweenness_ordered",
            Axiom::AngleMeasureExists => "angle_measure_exists",
            Axiom::AnglesSumTo180 => "angles_sum_to_180",
            Axiom::AnglesSumTo360 => "angles_sum_to_360",
            Axiom::AnglesSumBelow180 => "angles_sum_below_180",
            Axiom::DesarguesIncidence => "desargues_incidence",
            Axiom::LinesUnboundedBothWays => "lines_unbounded_both_ways",
            Axiom::IncidenceAxiom => "incidence_axiom",
            Axiom::MetricDefined => "metric_defined",
        }
    }

    /// Every axiom, as a slice, for a caller auditing the tables.
    pub fn all() -> &'static [Axiom] {
        &Axiom::ALL
    }

    /// The geometries that hold this axiom, derived from the tables rather
    /// than restated: a second hand-written list of the same facts is a
    /// second thing to get wrong.
    pub fn held_by(&self) -> Vec<GeometryKind> {
        GeometryKind::ALL.iter().copied().filter(|kind| kind.asserts(*self)).collect()
    }
}

impl fmt::Display for Axiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let claim = match self {
            Axiom::UniqueParallelThroughPoint => {
                "through a point off a line, exactly one line misses it"
            }
            Axiom::InfinitelyManyParallelsThroughPoint => {
                "through a point off a line, infinitely many lines miss it"
            }
            Axiom::NoUniqueParallel => "the unique-parallel claim fails here",
            Axiom::ParallelLinesNeverMeet => {
                "two distinct parallel lines share no point of the plane"
            }
            Axiom::AllLinesMeet => "any two distinct lines meet at exactly one point",
            Axiom::PointsAtInfinityExist => {
                "the completion has ideal points, one per direction"
            }
            Axiom::NoPointsAtInfinity => "the plane has no point at infinity",
            Axiom::LineAtInfinityExcluded => "the line at infinity is not a line of the plane",
            Axiom::CrossRatioInvariant => {
                "the cross ratio of four collinear points is a collineation invariant"
            }
            Axiom::BetweennessOrdered => {
                "a line carries a linear order, so one of three collinear points is between the other two"
            }
            Axiom::AngleMeasureExists => "angles carry a measure",
            Axiom::AnglesSumTo180 => "a triangle's angles sum to 180 degrees",
            Axiom::AnglesSumTo360 => {
                "a spherical triangle's angles sum to more than 180 degrees, and the angles around a point to 360"
            }
            Axiom::AnglesSumBelow180 => {
                "a hyperbolic triangle's angles sum to less than 180 degrees"
            }
            Axiom::DesarguesIncidence => {
                "triangles perspective from a point are perspective from a line"
            }
            Axiom::LinesUnboundedBothWays => "a line is unbounded in both directions",
            Axiom::IncidenceAxiom => "two distinct points determine exactly one line",
            Axiom::MetricDefined => "a finite distance is defined between any two points",
        };
        write!(f, "{claim}")
    }
}

// -------------------------------------------------------- axiom tables ---
//
// The tables are the declaration. They are `const` slices rather than data read
// at runtime because an environment that could change after a rule engine has
// read it would be an environment whose permits answer nothing, and the point
// is that a permit means the same thing on every call.

const EUCLIDEAN_AXIOMS: &[Axiom] = &[
    Axiom::UniqueParallelThroughPoint,
    Axiom::ParallelLinesNeverMeet,
    Axiom::NoPointsAtInfinity,
    Axiom::BetweennessOrdered,
    Axiom::AngleMeasureExists,
    Axiom::AnglesSumTo180,
    Axiom::DesarguesIncidence,
    Axiom::LinesUnboundedBothWays,
    Axiom::IncidenceAxiom,
    Axiom::MetricDefined,
];

const AFFINE_AXIOMS: &[Axiom] = &[
    Axiom::UniqueParallelThroughPoint,
    Axiom::ParallelLinesNeverMeet,
    Axiom::PointsAtInfinityExist,
    Axiom::LineAtInfinityExcluded,
    Axiom::CrossRatioInvariant,
    Axiom::BetweennessOrdered,
    Axiom::DesarguesIncidence,
    Axiom::IncidenceAxiom,
];

const PROJECTIVE_AXIOMS: &[Axiom] = &[
    Axiom::AllLinesMeet,
    Axiom::PointsAtInfinityExist,
    Axiom::CrossRatioInvariant,
    Axiom::DesarguesIncidence,
    Axiom::IncidenceAxiom,
];

const SPHERICAL_AXIOMS: &[Axiom] = &[
    Axiom::NoUniqueParallel,
    Axiom::AllLinesMeet,
    Axiom::NoPointsAtInfinity,
    Axiom::AngleMeasureExists,
    Axiom::AnglesSumTo360,
    Axiom::IncidenceAxiom,
    Axiom::MetricDefined,
];

const HYPERBOLIC_AXIOMS: &[Axiom] = &[
    Axiom::InfinitelyManyParallelsThroughPoint,
    Axiom::NoUniqueParallel,
    Axiom::PointsAtInfinityExist,
    Axiom::BetweennessOrdered,
    Axiom::AngleMeasureExists,
    Axiom::AnglesSumBelow180,
    Axiom::DesarguesIncidence,
    Axiom::LinesUnboundedBothWays,
    Axiom::IncidenceAxiom,
    Axiom::MetricDefined,
];

impl GeometryKind {
    /// The axioms this geometry validates, as a decision rather than a
    /// description: an axiom either holds in the declared environment or the
    /// environment does not have it, and a rule resting on an axiom the
    /// environment lacks is not permitted to fire.
    pub fn axioms(&self) -> &'static [Axiom] {
        match self {
            GeometryKind::Euclidean => EUCLIDEAN_AXIOMS,
            GeometryKind::Affine => AFFINE_AXIOMS,
            GeometryKind::Projective => PROJECTIVE_AXIOMS,
            GeometryKind::Spherical => SPHERICAL_AXIOMS,
            GeometryKind::Hyperbolic => HYPERBOLIC_AXIOMS,
        }
    }

    /// Whether this environment asserts `axiom`. The negative answers are the
    /// interesting ones and are what the refusals are built from: no geometry
    /// asserts everything, and `UniqueParallelThroughPoint` is false in four
    /// of the five.
    pub fn asserts(&self, axiom: Axiom) -> bool {
        self.axioms().contains(&axiom)
    }
}

// ------------------------------------------------------- theorem tables ---
//
// A theorem name is licensed by the axioms its *proof* needs, not by the
// conclusion. That is why several lists look generous: a theorem that survives
// every affine transformation is a theorem of the affine plane, even though
// the affine plane has no lengths and the theorem is usually stated in terms
// of them. And it is why the Euclidean list carries the kernel's own rule
// names: that library was written under Euclid, and this is the list that lets
// an environment gate it honestly instead of assuming it.

const EUCLIDEAN_THEOREMS: &[&str] = &[
    "CollinearTransitivity",
    "ParallelTransitivity",
    "PerpendicularTransitivity",
    "ParallelPerpendicular",
    "MidpointCollinear",
    "MidpointBisectsPerpendicular",
    "AngleIsosceles",
    "EqualAnglesSubtendArc",
    "ParallelogramDiagonals",
    "triangle_angle_sum",
    "alternate_angles_equal",
    "consecutive_interior_angles_supplementary",
    "unique_parallel_through_point",
    "midpoint_theorem",
    "isosceles_base_angles_equal",
    "exterior_angle_greater_than_opposite",
    "pythagoras",
    "thales_diameter_right_angle",
    "similar_triangles_by_aaa",
];

const AFFINE_THEOREMS: &[&str] = &[
    "CollinearTransitivity",
    "ParallelTransitivity",
    "MidpointCollinear",
    "ParallelogramDiagonals",
    "intercept_theorem",
    "affine_combination_of_points",
    "parallel_lines_meet_at_infinity",
    "unique_parallel_through_point",
    "desargues",
    "ratios_preserved_by_affine_maps",
];

const PROJECTIVE_THEOREMS: &[&str] = &[
    "CollinearTransitivity",
    "all_lines_meet",
    "cross_ratio_invariant",
    "cross_ratio_harmonic_conjugate",
    "harmonic_range_from_a_complete_quadrangle",
    "desargues",
    "two_lines_meet_exactly_once",
    "projective_frame_maps_to_any_frame",
];

const SPHERICAL_THEOREMS: &[&str] = &[
    "CollinearTransitivity",
    "all_lines_meet",
    "great_circles_meet_in_two_antipodal_points",
    "antipodal_points_define_the_same_line",
    "angles_around_a_point_sum_to_360",
    "spherical_excess_equals_area",
    "polar_line_of_a_point",
    "spherical_triangle_has_three_sides",
    "verse_sine_law_angles",
];

const HYPERBOLIC_THEOREMS: &[&str] = &[
    "CollinearTransitivity",
    "infinitely_many_parallels_through_point",
    "existence_of_asymptotic_parallel",
    "existence_of_ultraparallel",
    "cross_ratio_invariant",
    "betweenness_ordered",
    "desargues",
    "hyperbolic_triangle_angle_defect",
    "midpoint_exists_uniquely",
    "lines_unbounded_both_ways",
];

impl GeometryKind {
    /// The theorems and rules this geometry licenses the engine to fire.
    ///
    /// The Euclidean list carries the kernel's own rule names, because that
    /// library was written under Euclid. The other lists carry the theorems
    /// whose *proofs* need the axioms above: Desargues needs a Desarguesian
    /// plane, cross-ratio invariance needs projective incidence, and the
    /// hyperbolic many-parallels statement needs the absolute conic.
    pub fn valid_theorems(&self) -> &'static [&'static str] {
        match self {
            GeometryKind::Euclidean => EUCLIDEAN_THEOREMS,
            GeometryKind::Affine => AFFINE_THEOREMS,
            GeometryKind::Projective => PROJECTIVE_THEOREMS,
            GeometryKind::Spherical => SPHERICAL_THEOREMS,
            GeometryKind::Hyperbolic => HYPERBOLIC_THEOREMS,
        }
    }

    /// Whether `rule_or_theorem` may be fired in this environment. Names are
    /// matched exactly: a near-miss is a rule this environment does not
    /// license, and quietly accepting a misspelled axiom dependency is how a
    /// proof ends up resting on nothing.
    pub fn permits(&self, rule_or_theorem: &str) -> bool {
        self.valid_theorems().contains(&rule_or_theorem)
    }

    /// License a rule, or refuse it by name with the permitted list attached.
    /// The `Err` arm is the point of the function: a rule engine that only
    /// received `false` would have to invent what to do about it, and an error
    /// that names the theorems it *would* license is a decision it can make.
    pub fn require_theorem(&self, rule_or_theorem: &str) -> anyhow::Result<&'static str> {
        match self.valid_theorems().iter().find(|name| **name == rule_or_theorem) {
            Some(licensed) => Ok(*licensed),
            None => Err(NonEuclideanError::TheoremNotPermitted {
                geometry: *self,
                theorem: rule_or_theorem.to_string(),
                permitted: self.valid_theorems().iter().map(|name| name.to_string()).collect(),
            }
            .into()),
        }
    }
}

// ------------------------------------------------- exactness vs evidence ---

/// How far a float-based claim is from being a fact, in the one place a number
/// of degrees is compared with an axiom.
///
/// Half a degree is far too loose to prove anything and tight enough that a
/// rounding error cannot manufacture agreement: a genuine 180-degree sum
/// computed from three `acos`es lands within `1e-13` of 180, and a
/// 179.6-degree hyperbolic sum is a real disagreement rather than a rounding
/// artifact. The comparison is reported as *consistency*, never as proof -- the
/// sum of three angles is not a rational, which is the entire reason this
/// constant lives beside a [`NumericEvidence`] type instead of a [`Fact`].
const ANGLE_SUM_TOLERANCE_DEGREES: f64 = 0.5;

/// An exact incidence verdict: did these two lines meet, and where.
///
/// The `Err` arms are refusals (a degenerate line, a coincident pair); the
/// `Ok` arms are decisions. There is deliberately no third "probably" and no
/// floating point: over the models below, meeting is a polynomial test in
/// exact rationals, and a model that could only estimate it would be a model
/// whose estimates get quoted as results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Incidence {
    /// The two lines meet at this exact point, inside the plane. A projective
    /// point with `z == 1` is a finite affine point; the spherical model
    /// reports a direction, which is the same object with no affine chart.
    Meet(PPoint),
    /// The two lines share no point of the plane, but they meet in the
    /// completion at this exact ideal point. In the affine plane this is
    /// what "parallel" means; in the projective plane it is all that a
    /// parallel pair is.
    MeetAtInfinity(PPoint),
    /// The two lines share no point at all, not even an ideal one. Only the
    /// hyperbolic model has this arm, and its existence is the axiom that
    /// separates that plane from Euclid: here a "parallel" claim is not even
    /// the right shape, because the lines might have met at an ideal point or
    /// nowhere at all.
    Disjoint,
}

impl Incidence {
    /// Whether the two lines met inside the plane of the model, exactly.
    pub fn meets(&self) -> bool {
        matches!(self, Incidence::Meet(_))
    }

    /// The exact point where the two lines meet, finite or ideal. `None` for
    /// a genuinely disjoint pair -- a hyperbolic ultraparallel, which is a
    /// decision too.
    pub fn at(&self) -> Option<&PPoint> {
        match self {
            Incidence::Meet(point) | Incidence::MeetAtInfinity(point) => Some(point),
            Incidence::Disjoint => None,
        }
    }
}

impl fmt::Display for Incidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Incidence::Meet(point) => write!(f, "they meet at {}", point.describe()),
            Incidence::MeetAtInfinity(point) => {
                write!(f, "they meet only at the ideal point {}", point.describe())
            }
            Incidence::Disjoint => write!(f, "they share no point, ideal or finite"),
        }
    }
}

/// A number that was computed and is *not* a fact.
///
/// The kernel's own rule for a diagram that merely looks parallel applies here,
/// and for a sharper reason: the sum of three angles is not a rational number
/// at all. `cos` is exact ([`QSqrt`]) but `acos` is not, so "the angles of this
/// triangle sum to 180" can only ever be *measured* in this module. A type that
/// carries the value and refuses to become a [`Fact`] is what stops that
/// measurement from being quoted as a derivation -- the same discipline as
/// `Fact::observed` versus `Fact::derived`, with the confidence field replaced
/// by a refusal, because there is no useful confidence to attach to a quantity
/// that has no exact value to be nearly equal to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericEvidence {
    /// What was measured, in words: `"angle sum of the angles at A, B, C"`.
    pub quantity: String,
    /// The measured value, in the unit named by [`Self::unit`].
    pub value: f64,
    /// The unit, so a bare number never has to be guessed at.
    pub unit: String,
    /// Why this is evidence and not a fact, carried with the number so a
    /// report that quotes the value quotes the caveat too.
    pub caveat: String,
}

impl NumericEvidence {
    /// Evidence in degrees.
    pub fn degrees(quantity: &str, value: f64) -> Self {
        Self {
            quantity: quantity.to_string(),
            value,
            unit: "degrees".to_string(),
            caveat: "measured by a numeric method; not a fact and not derivable".to_string(),
        }
    }

    /// Evidence as a plain real number with a named unit.
    pub fn of(quantity: &str, value: f64, unit: &str) -> Self {
        Self {
            quantity: quantity.to_string(),
            value,
            unit: unit.to_string(),
            caveat: "measured by a numeric method; not a fact and not derivable".to_string(),
        }
    }

    /// The sum of three angles, in degrees, from their exact cosines. The
    /// cosines are exact inputs; the sum is the first approximation in this
    /// module, and the return type is what says so.
    pub fn angle_sum_degrees(angles: &[Angle3], cosines: &[QSqrt]) -> anyhow::Result<Self> {
        anyhow::ensure!(
            angles.len() == cosines.len() && !cosines.is_empty(),
            "an angle sum needs one exact cosine per angle: {} angles, {} cosines",
            angles.len(),
            cosines.len()
        );
        let mut total = 0.0f64;
        for cos in cosines {
            let value = cos.to_f64()?;
            anyhow::ensure!(
                (-1.0..=1.0).contains(&value),
                "a cosine of {value} is not a cosine: the exact radical must have been \
                 mispaired with its radicand before it got here"
            );
            total += value.clamp(-1.0, 1.0).acos().to_degrees();
        }
        let named = angles.iter().map(|angle| angle.at.as_str()).collect::<Vec<&str>>().join(", ");
        Ok(Self::degrees(&format!("angle sum of the angles at {named}"), total))
    }

    /// Refuse to promote this to a fact. Always an `Err`; the `Ok` type is
    /// never produced, which is the enforcement the honesty rule asks for. A
    /// caller that wants a fact has to find an exact route to one.
    pub fn into_fact(self) -> anyhow::Result<Fact> {
        Err(NonEuclideanError::EvidenceIsNotAFact { quantity: self.quantity }.into())
    }

    /// Whether the declared geometry's angle-sum axiom is *consistent with*
    /// this measurement -- a note, not a proof. `None` where the geometry
    /// declares no angle-sum axiom, and `None` for a quantity that is not in
    /// degrees, because there is no axiom to be consistent with.
    pub fn agrees_with(&self, kind: GeometryKind) -> Option<bool> {
        match self.unit.as_str() {
            "degrees" => kind.agrees_with_angle_sum(self.value),
            _ => None,
        }
    }
}

impl fmt::Display for NumericEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {} {} [evidence: {}]", self.quantity, self.value, self.unit, self.caveat)
    }
}

/// Hyperbolic trigonometry, refused.
///
/// The hyperbolic plane's distance and angle formulas need `sinh` and `cosh` of
/// a hyperbolic distance, and those live in no field this kernel has: not
/// `Q`, and not `Q(sqrt d)` either, since the hyperbolic cosine of a rational
/// length is generally not algebraic of degree two at all. Returning a `f64`
/// here would put an approximation exactly where the kernel's discipline is
/// that there is none -- the same sin the geometry audit found was never the
/// ambition, only the prose that ran ahead of it. So this refuses, with the
/// reason attached, and the hyperbolic claims this module *can* prove (the
/// incidence ones) are proved exactly instead.
pub fn hyperbolic_trig(operation: &str) -> anyhow::Result<QSqrt> {
    Err(NonEuclideanError::OutOfScope {
        detail: format!(
            "hyperbolic {operation}: the hyperbolic sine and cosine of a rational length are in \
             neither the rationals nor the quadratic field the kernel computes in, so an exact \
             answer has no representation here, and an approximate one would be a float where a \
             fact is claimed"
        ),
    }
    .into())
}

// ------------------------------------------------------- checking a scene ---

/// A statement the declared geometry cannot accept.
///
/// The shape matters: the offending *statement*, the *axiom* it contradicts,
/// and a one-line reason naming the exact witness that settles it. A violation
/// without that third part is only useful to someone who goes and recomputes
/// it, and the recomputation is the thing that has to not happen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxiomViolation {
    /// The axiom the statement contradicts.
    pub axiom: Axiom,
    /// The statement as the scene states it, through the kernel's own
    /// `describe`, so the two modules agree on how a constraint reads.
    pub statement: String,
    /// Why it is impossible here, with the exact witness where one exists.
    pub detail: String,
}

/// A statement the declared geometry cannot decide.
///
/// Separate from a violation because the two call for opposite responses. A
/// violation is a wrong fact to remove; an underdetermination is a fact that is
/// not specific enough to reason from -- the hyperbolic `Parallel`, which is
/// true of a great many line pairs and does not say which kind. Reporting the
/// second as the first would train a reasoner to delete the true statements of
/// a non-Euclidean figure, which is the same over-application wearing a
/// different hat.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Underdetermination {
    /// The axiom whose *absence* leaves the statement open.
    pub axiom: Axiom,
    /// The statement as the scene states it.
    pub statement: String,
    /// What the environment would need in order to decide it.
    pub needed: String,
}

/// The result of checking a scene against a declared geometry: the axioms that
/// were checked, the statements that contradict them, and the statements they
/// leave open.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxiomReport {
    /// The environment the scene was read under.
    pub geometry: GeometryKind,
    /// Every axiom of that environment, so a reader of the report has the
    /// whole environment and not just the parts that were upset.
    pub axioms: Vec<Axiom>,
    /// Statements the environment cannot accept.
    pub violations: Vec<AxiomViolation>,
    /// Statements the environment cannot decide.
    pub underdetermined: Vec<Underdetermination>,
    /// The statements that were checked, for a report that can say what it
    /// looked at.
    pub checked: Vec<String>,
}

impl AxiomReport {
    /// Whether the scene is consistent with its declared geometry. A scene
    /// with underdetermined statements is consistent -- it just is not yet
    /// specific enough to reason from -- so the two questions stay apart.
    pub fn is_consistent(&self) -> bool {
        self.violations.is_empty()
    }

    /// Whether every statement was decided, which is what a rule engine needs
    /// before it fires: an undecided statement is not a licence.
    pub fn is_complete(&self) -> bool {
        self.underdetermined.is_empty()
    }

    /// The violation naming `statement`, or `None`.
    pub fn violation_for(&self, statement: &str) -> Option<&AxiomViolation> {
        self.violations.iter().find(|violation| violation.statement == statement)
    }

    /// The underdetermination naming `statement`, or `None`.
    pub fn underdetermination_for(&self, statement: &str) -> Option<&Underdetermination> {
        self.underdetermined.iter().find(|open| open.statement == statement)
    }
}

impl fmt::Display for AxiomReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} checked, {} violation(s), {} underdetermined",
            self.geometry,
            self.checked.len(),
            self.violations.len(),
            self.underdetermined.len()
        )?;
        for violation in &self.violations {
            write!(
                f,
                "\n  violation: {} -- {} ({})",
                violation.statement, violation.detail, violation.axiom
            )?;
        }
        for open in &self.underdetermined {
            write!(f, "\n  undecided: {} -- needs {} ({})", open.statement, open.needed, open.axiom)?;
        }
        Ok(())
    }
}

impl GeometryKind {
    /// Check a scene against this declared geometry.
    ///
    /// Every fact in the ledger is read as a *claim about this environment*,
    /// and each claim is routed to the axiom whose absence or presence decides
    /// it. Three outcomes per claim, and the choice between the last two is the
    /// substance of the exercise:
    ///
    /// - decided and fine: the claim is a statement of this environment;
    /// - [`AxiomViolation`]: the claim is *false* here -- two lines marked
    ///   `Parallel` in the projective plane, where every two lines meet;
    /// - [`Underdetermination`]: the claim has no truth value here -- a
    ///   `Parallel` in the hyperbolic plane, which is under-specified rather
    ///   than wrong, and `Between` projectively, which names an order the
    ///   projective line does not have.
    ///
    /// Facts the kernel recorded below full confidence are *not* checked:
    /// a diagram's guess that two strokes look parallel is evidence, and this
    /// report does not get to call it a contradiction. That is the kernel's
    /// `Fact::observed` discipline, and it is why an axiom check over a
    /// diagram-graded scene reports nothing rather than reporting noise.
    pub fn check_axioms(&self, graph: &SceneGraph) -> anyhow::Result<AxiomReport> {
        self.check_scene(graph)?;
        let mut report = AxiomReport {
            geometry: *self,
            axioms: self.axioms().to_vec(),
            violations: Vec::new(),
            underdetermined: Vec::new(),
            checked: Vec::new(),
        };
        for fact in &graph.facts {
            if !fact.is_established() {
                continue;
            }
            self.check_fact(graph, &fact.constraint, &mut report)?;
        }
        Ok(report)
    }

    /// Route one statement to the axiom that decides it.
    fn check_fact(
        &self,
        graph: &SceneGraph,
        constraint: &Constraint,
        report: &mut AxiomReport,
    ) -> anyhow::Result<()> {
        let statement = constraint.describe();
        match constraint {
            // The parallel case is the one the audit is about, so it is
            // decided by the geometry's own model rather than by a table.
            Constraint::Parallel { first, second } => {
                if self.asserts(Axiom::ParallelLinesNeverMeet) {
                    report.checked.push(statement);
                    return Ok(());
                }
                if self.asserts(Axiom::AllLinesMeet) {
                    let detail = self.meeting_witness(graph, first, second);
                    report.violations.push(AxiomViolation {
                        axiom: Axiom::AllLinesMeet,
                        statement,
                        detail: format!(
                            "{self} is a geometry where any two distinct lines meet, so a \
                             parallel pair is a contradiction: {detail}"
                        ),
                    });
                } else {
                    report.underdetermined.push(Underdetermination {
                        axiom: Axiom::InfinitelyManyParallelsThroughPoint,
                        statement,
                        needed: "which kind of non-meeting the pair stands in: asymptotic \
                                 (sharing an ideal point) or ultraparallel (sharing nothing). \
                                 'Parallel' names both and so decides neither."
                            .to_string(),
                    });
                }
            }
            // A metric claim in a geometry with no metric. Not a violation of
            // an axiom -- the statement is false of the *geometry*, not
            // contradicted by it, since an affine transformation of the figure
            // changes the quantity while staying in the plane. Reported as a
            // violation because accepting it would put a Euclidean assumption
            // into a geometry that refuses it.
            Constraint::EqualLength { .. }
            | Constraint::LengthIs { .. }
            | Constraint::ScaleLength { .. }
            | Constraint::AreaEqual { .. }
            | Constraint::Congruent { .. }
            | Constraint::Circle { .. } => {
                if self.asserts(Axiom::MetricDefined) {
                    report.checked.push(statement);
                } else {
                    report.violations.push(AxiomViolation {
                        axiom: Axiom::MetricDefined,
                        statement,
                        detail: format!(
                            "{self} has no metric: lengths, areas and congruence are not \
                             preserved by its own transformations, so this quantity belongs to \
                             the drawing and not to the plane"
                        ),
                    });
                }
            }
            // Angles: a measurement claim. Refused in the affine and
            // projective planes, whose transformation groups do not preserve
            // angle measure.
            Constraint::RightAngle { .. }
            | Constraint::AngleEqual { .. }
            | Constraint::AngleIs { .. } => {
                if self.asserts(Axiom::AngleMeasureExists) {
                    report.checked.push(statement);
                } else {
                    report.violations.push(AxiomViolation {
                        axiom: Axiom::AngleMeasureExists,
                        statement,
                        detail: format!(
                            "{self} has no angle measure, so an angle is not one of its \
                             quantities: the right statement to make about a figure here is an \
                             incidence one, or a cross ratio"
                        ),
                    });
                }
            }
            // Order along a line. A violation, not an underdetermination: a
            // projective line is a circle and a great circle is closed, so
            // "between" names a relation these lines simply do not have, and
            // the ratio of two collinear segments needs the point at infinity
            // as a unit.
            Constraint::Between { .. } | Constraint::MidpointOf { .. } => {
                if self.asserts(Axiom::BetweennessOrdered) {
                    report.checked.push(statement);
                } else {
                    report.violations.push(AxiomViolation {
                        axiom: Axiom::BetweennessOrdered,
                        statement,
                        detail: format!(
                            "{self} has no linear order along a line, so one of three collinear \
                             points is not between the other two: 'between' and a midpoint name a \
                             relation this geometry does not define"
                        ),
                    });
                }
            }
            // Everything else is incidence, which every one of these
            // geometries shares (that is what `IncidenceAxiom` records).
            _ => report.checked.push(statement),
        }
        Ok(())
    }

    /// The exact witness that two segments do in fact meet, for the geometries
    /// that say they must. Lifted into the declared model, so the witness is a
    /// point of *that* geometry rather than a claim about the drawing: in the
    /// projective plane the meeting point is the ideal point of the common
    /// direction, and on the sphere the two great circles through the
    /// segments cross in two antipodal directions.
    fn meeting_witness(
        &self,
        graph: &SceneGraph,
        first: &Segment,
        second: &Segment,
    ) -> String {
        let (p1, p2) = match (ProjectivePlane::segment_line(graph, first), ProjectivePlane::segment_line(graph, second)) {
            (Ok(one), Ok(other)) => (one, other),
            _ => return "the exact witness could not be built: a segment is degenerate or names an undeclared point".to_string(),
        };
        match *self {
            GeometryKind::Projective => match p1.meet(&p2) {
                Ok(where_) => format!("they meet at the exact point {}", where_.describe()),
                Err(_) => "the exact witness could not be built: the two segments are the same line".to_string(),
            },
            GeometryKind::Spherical => {
                match SphericalModel::meet_of_segments(graph, first, second) {
                    Ok(ray) => {
                        format!("the two great circles through the segments meet in the exact direction {}", ray.describe())
                    }
                    Err(_) => "the two segments name one great circle, so they are the same line, not two".to_string(),
                }
            }
            _ => "the declared geometry decides this without a witness".to_string(),
        }
    }
}

/// Serde for [`Q`], which the kernel deliberately leaves without `Serialize`:
/// its own file format uses [`Frac`] for a rational, so that a coordinate in a
/// scene file is a `num`/`den` pair and never a float.
///
/// These types keep their arithmetic in [`Q`] -- every operation on a
/// homogeneous point is a rational operation, and storing `Frac` would mean
/// converting on every line -- so the conversion lives here, at the boundary,
/// where it is one function instead of a hundred call sites. The shape on the
/// wire is the kernel's, so a projective point and a kernel point serialize
/// the same way and a scene file has one rational representation throughout.
mod q_serde {
    use super::{Frac, Q};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(value: &Q, serializer: S) -> Result<S::Ok, S::Error> {
        Frac::from_q(*value).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Q, D::Error> {
        Frac::deserialize(deserializer)?
            .to_q()
            .map_err(serde::de::Error::custom)
    }
}

// ------------------------------------------------- the projective plane ---

/// A point of the projective plane, in exact homogeneous coordinates.
///
/// `[x : y : z]` is a *direction-with-position*: scaling the triple by any
/// nonzero rational gives the same point, which is exactly why the projective
/// plane can hold the point at infinity that the affine plane needs but cannot
/// contain. A point with `z == 0` is at infinity and has no affine
/// coordinates; [`Self::to_affine`] refuses it rather than inventing a chart,
/// because the choice of which points are finite is the choice that defines
/// the affine plane, and this module refuses to make it silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PPoint {
    #[serde(with = "q_serde")]
    pub x: Q,
    #[serde(with = "q_serde")]
    pub y: Q,
    #[serde(with = "q_serde")]
    pub z: Q,
}

impl PPoint {
    /// A homogeneous point, refusing the zero triple. `(0, 0, 0)` is not a
    /// point at infinity, it is nothing: every scaling of it is still nothing.
    pub fn new(x: Q, y: Q, z: Q) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !(x.is_zero() && y.is_zero() && z.is_zero()),
            "{}",
            GeometryError::EmptyGeometry("the homogeneous triple (0, 0, 0) is not a point: it \
                 has no direction and no position, and every nonzero scaling of it is still \
                 zero"
                .to_string())
        );
        Ok(Self { x, y, z })
    }

    /// The finite point `(x, y, 1)`.
    pub fn affine(x: Q, y: Q) -> Self {
        Self { x, y, z: Q::ONE }
    }

    /// The point at infinity of a direction, `[dx : dy : 0]`, refusing a zero
    /// direction for the same reason [`Self::new`] refuses the zero triple.
    pub fn at_infinity(dx: Q, dy: Q) -> anyhow::Result<Self> {
        Self::new(dx, dy, Q::ZERO)
    }

    /// Whether this point is at infinity, exactly.
    pub fn is_at_infinity(&self) -> bool {
        self.z.is_zero()
    }

    /// The affine coordinates, refusing an ideal point. The error names the
    /// remedy rather than the failure: this is a projective figure, and the
    /// affine chart is a choice the caller has to make explicit.
    pub fn to_affine(&self) -> anyhow::Result<(Q, Q)> {
        anyhow::ensure!(
            !self.is_at_infinity(),
            "{}",
            NonEuclideanError::OutOfScope {
                detail: format!(
                    "{self} is a point at infinity and has no affine coordinates; the \
                     projective plane has no preferred line at infinity, so a chart has to \
                     be chosen before this point means a place"
                ),
            }
        );
        // Dividing by `z` is the point of the conversion: a homogeneous triple
        // is a point only up to a nonzero scale, so the cross product of two
        // lines is some `[4 : 2 : 4]` where the point is `(1, 1/2)`. Returning
        // the raw pair would be a silent factor-of-`z` error that only shows up
        // when the arithmetic happens to be integral.
        Ok((self.x.div(&self.z)?, self.y.div(&self.z)?))
    }

    /// Scaled by a nonzero rational, which is the same point.
    pub fn scale(&self, factor: &Q) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !factor.is_zero(),
            "{}",
            GeometryError::EmptyGeometry("scaling a projective point by zero erases it".to_string())
        );
        Ok(Self { x: self.x.mul(factor)?, y: self.y.mul(factor)?, z: self.z.mul(factor)? })
    }

    /// A normalized representative: the first nonzero coordinate is `1`.
    /// Needed for comparing and printing, since `PPoint` equality is equality
    /// of triples and `[1:2:3]` is not literally `[-1:-2:-3]`.
    pub fn normalized(&self) -> anyhow::Result<Self> {
        if !self.x.is_zero() {
            Self::new(Q::ONE, self.y.div(&self.x)?, self.z.div(&self.x)?)
        } else if !self.y.is_zero() {
            Self::new(Q::ZERO, Q::ONE, self.z.div(&self.y)?)
        } else {
            anyhow::ensure!(!self.z.is_zero(), "{}", GeometryError::EmptyGeometry("the zero point does not normalize".to_string()));
            Self::new(Q::ZERO, Q::ZERO, Q::ONE)
        }
    }

    /// The point as the kernel's own point type, refusing an ideal point: a
    /// [`KPoint`] has two coordinates and so is an affine object.
    pub fn to_kpoint(&self, name: &str) -> anyhow::Result<KPoint> {
        let (x, y) = self.to_affine()?;
        Ok(KPoint { name: name.to_string(), x: Frac::from_q(x), y: Frac::from_q(y) })
    }
}

impl fmt::Display for PPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{} : {} : {}]", self.x, self.y, self.z)
    }
}

impl PPoint {
    /// A one-line identity for reports, normalized so that two spellings of
    /// one point print the same.
    pub fn describe(&self) -> String {
        match self.normalized() {
            Ok(point) => point.to_string(),
            Err(_) => self.to_string(),
        }
    }
}

/// A line of the projective plane, in exact coefficients: the set of `p` with
/// `a x + b y + c z == 0`.
///
/// The coefficients are projective too -- scaling `(a, b, c)` names the same
/// line -- and the same non-degeneracy rules apply: the zero triple is not a
/// line, it is every point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PLine {
    #[serde(with = "q_serde")]
    pub a: Q,
    #[serde(with = "q_serde")]
    pub b: Q,
    #[serde(with = "q_serde")]
    pub c: Q,
}

impl PLine {
    /// A line by its coefficients, refusing the zero triple.
    pub fn new(a: Q, b: Q, c: Q) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !(a.is_zero() && b.is_zero() && c.is_zero()),
            "{}",
            GeometryError::EmptyGeometry("the coefficient triple (0, 0, 0) is not a line: it \
                 contains every point, so it constrains nothing"
                .to_string())
        );
        Ok(Self { a, b, c })
    }

    /// The line through two points, refusing coincident ones. In the
    /// projective plane this is the *only* way to get a line from data, and
    /// two identical points do not determine one -- which is the first place
    /// the projective axioms bite, since "distinct" is not automatic.
    pub fn through(p: &PPoint, q: &PPoint) -> anyhow::Result<Self> {
        // the cross product of the two triples is perpendicular to both
        let (a, b, c) = (
            p.y.mul(&q.z)?.sub(&p.z.mul(&q.y)?)?,
            p.z.mul(&q.x)?.sub(&p.x.mul(&q.z)?)?,
            p.x.mul(&q.y)?.sub(&p.y.mul(&q.x)?)?,
        );
        Self::new(a, b, c)
    }

    /// The line at infinity, `z == 0`.
    pub fn line_at_infinity() -> Self {
        // The coefficients are nonzero by construction, so this is total: the
        // all-zero refusal in `new` cannot fire for a constant.
        Self { a: Q::ZERO, b: Q::ZERO, c: Q::ONE }
    }

    /// Whether `point` lies on this line, exactly.
    pub fn contains(&self, point: &PPoint) -> anyhow::Result<bool> {
        let value = self
            .a
            .mul(&point.x)?
            .add(&self.b.mul(&point.y)?)?
            .add(&self.c.mul(&point.z)?)?;
        Ok(value.is_zero())
    }

    /// The exact point where two lines meet, refusing coincident ones. This is
    /// the `AllLinesMeet` axiom as code: over the reals the cross product of
    /// two non-proportional lines is never zero, so the two always meet and
    /// the only refusal is a pair that was never two lines.
    pub fn meet(&self, other: &PLine) -> anyhow::Result<PPoint> {
        let (x, y, z) = (
            self.b.mul(&other.c)?.sub(&self.c.mul(&other.b)?)?,
            self.c.mul(&other.a)?.sub(&self.a.mul(&other.c)?)?,
            self.a.mul(&other.b)?.sub(&self.b.mul(&other.a)?)?,
        );
        PPoint::new(x, y, z).map_err(|_| {
            GeometryError::CoincidentLines.into()
        })
    }

    /// The point at infinity of this line, `[b : -a : 0]`: the direction the
    /// line runs in. Two affine lines are parallel exactly when these agree.
    /// A line that *is* the line at infinity has no single direction, and is
    /// refused rather than given an arbitrary one.
    pub fn at_infinity(&self) -> anyhow::Result<PPoint> {
        anyhow::ensure!(
            !self.a.is_zero() || !self.b.is_zero(),
            "{}",
            GeometryError::DegenerateSegment
        );
        PPoint::new(self.b, self.a.neg()?, Q::ZERO)
    }

    /// Whether two lines share a point at infinity, which in the affine chart
    /// is exactly "parallel". In the projective plane it is a statement about
    /// the *completion* of the affine figure, not about the lines meeting.
    pub fn meets_at_infinity_of(&self, other: &PLine) -> anyhow::Result<bool> {
        let (mine, theirs) = (self.at_infinity()?, other.at_infinity()?);
        // equal directions iff the cross product vanishes; compared through a
        // ratio rather than a triple, since both are at infinity
        Ok(mine.x.mul(&theirs.y)?.sub(&mine.y.mul(&theirs.x)?)?.is_zero())
    }

    /// A one-line identity for reports, normalized.
    pub fn describe(&self) -> String {
        match self.normalized() {
            Ok(line) => line.to_string(),
            Err(_) => format!("[{} : {} : {}]", self.a, self.b, self.c),
        }
    }

    /// A normalized representative: the first nonzero coefficient is `1`.
    pub fn normalized(&self) -> anyhow::Result<Self> {
        if !self.a.is_zero() {
            Self::new(Q::ONE, self.b.div(&self.a)?, self.c.div(&self.a)?)
        } else if !self.b.is_zero() {
            Self::new(Q::ZERO, Q::ONE, self.c.div(&self.b)?)
        } else {
            anyhow::ensure!(!self.c.is_zero(), "{}", GeometryError::EmptyGeometry("the zero line does not normalize".to_string()));
            Self::new(Q::ZERO, Q::ZERO, Q::ONE)
        }
    }
}

impl fmt::Display for PLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{} : {} : {}]", self.a, self.b, self.c)
    }
}

/// The real projective plane: homogeneous coordinates, exact incidence, and
/// the exact cross ratio.
///
/// The cross ratio is the reason this model earns its place. Length and angle
/// are properties of a *metric*, and the projective plane has no metric: a
/// projectivity can stretch one direction without limit while fixing a line,
/// so no ratio of lengths and no angle survives. The cross ratio of four
/// collinear points does survive, is a rational, and is computed here in
/// exact arithmetic -- so a reasoner in this geometry has one number it can
/// decide, and it is not a number about lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectivePlane;

impl ProjectivePlane {
    /// The exact cross ratio `(A, B; C, D)` of four collinear points.
    ///
    /// Defined by projecting onto the line's own parameter: with `A` at `0` and
    /// `B` at `1`, the value is `(c / d) * ((d - 1) / (c - 1))` where `c` and
    /// `d` are the coordinates of `C` and `D`. The alternative conventions
    /// differ by which pairs are inverted; this one is the
    /// `(A,B;C,D) = AC/BC : AD/BD` form, and the harmonic quadruple
    /// `(0, 1, 1/2, infinity)` gives `-1` under it, which is the test that
    /// pins the convention down.
    ///
    /// All four points must be collinear and pairwise distinct. Coincidence is
    /// refused rather than simplified: the cross ratio of a degenerate
    /// quadruple is a limit, and a limit is not a fact about the figure.
    pub fn cross_ratio(a: &PPoint, b: &PPoint, c: &PPoint, d: &PPoint) -> anyhow::Result<Q> {
        let line = PLine::through(a, b)?;
        for other in [c, d] {
            anyhow::ensure!(
                line.contains(other)?,
                "{}",
                NonEuclideanError::DegenerateCrossRatio {
                    detail: format!("{} is not on the line through {} and {}", other.describe(), a.describe(), b.describe()),
                }
            );
        }
        // Distinctness, decided exactly rather than assumed.
        for (left, right) in [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)] {
            anyhow::ensure!(
                !same_point(left, right)?,
                "{}",
                NonEuclideanError::DegenerateCrossRatio {
                    detail: format!(
                        "the quadruple is not four distinct points: {} and {} are one point",
                        left.describe(),
                        right.describe()
                    ),
                }
            );
        }
        // The determinant form, which is exact in homogeneous coordinates and
        // so is the *same* formula whether or not one of the four points is
        // ideal. Reading off an affine parameter instead would divide by zero
        // on exactly the quadruple the projective plane exists for -- the
        // harmonic range with a point at infinity in it.
        //
        //   (A,B;C,D) = d(C,A) d(B,D) / ( d(C,B) d(A,D) )
        //
        // where `d(P, Q)` is the 2x2 determinant of any one coordinate pair,
        // taken in the order that keeps the sign. A pair on which every
        // determinant vanishes (the four points on a line parallel to that
        // coordinate plane) is not a reason to refuse: the next pair is tried,
        // and only when all three degenerate is the quadruple undecidable.
        let coordinate_pairs = [(0usize, 1usize), (0, 2), (1, 2)];
        let at = |p: &PPoint, index: usize| -> Q {
            match index {
                0 => p.x,
                1 => p.y,
                _ => p.z,
            }
        };
        for (i, j) in coordinate_pairs {
            let det = |p: &PPoint, r: &PPoint| -> anyhow::Result<Q> {
                at(p, i).mul(&at(r, j))?.sub(&at(p, j).mul(&at(r, i))?)
            };
            let (ca, bd, cb, ad) = (det(c, a)?, det(b, d)?, det(c, b)?, det(a, d)?);
            let denominator = cb.mul(&ad)?;
            if !denominator.is_zero() {
                return ca.mul(&bd)?.div(&denominator);
            }
        }
        Err(NonEuclideanError::DegenerateCrossRatio {
            detail: "the four points share no coordinate plane in which they can be                      distinguished, so no determinant pairing decides their cross ratio"
                .to_string(),
        }
        .into())
    }

    /// The fourth point of the harmonic quadruple `(A, B; C, D)` with
    /// cross ratio `-1`, which is the projective replacement for "D is the
    /// mirror of C across the midpoint of AB" -- the same construction the
    /// kernel's reflection does in the plane, carried where lengths do not
    /// exist.
    pub fn harmonic_conjugate(a: &PPoint, b: &PPoint, c: &PPoint) -> anyhow::Result<PPoint> {
        let (a_aff, b_aff, c_aff) = (a.to_affine()?, b.to_affine()?, c.to_affine()?);
        // in the parameter with A at 0, B at 1: t_D = t_C / (2 t_C - 1)
        let t = if !b_aff.0.sub(&a_aff.0)?.is_zero() {
            c_aff.0.sub(&a_aff.0)?.div(&b_aff.0.sub(&a_aff.0)?)?
        } else {
            anyhow::ensure!(!b_aff.1.sub(&a_aff.1)?.is_zero(), "{}", GeometryError::DegenerateSegment);
            c_aff.1.sub(&a_aff.1)?.div(&b_aff.1.sub(&a_aff.1)?)?
        };
        let den = t.mul(&Q::from_int(2))?.sub(&Q::ONE)?;
        // `2t - 1` vanishes exactly when C is the midpoint of AB -- and then the
        // harmonic conjugate is the point at *infinity* of the line, which is
        // the whole reason projective geometry has one. It is returned, not
        // refused: refusing the one configuration the construction exists for
        // would be a refusal of the answer rather than of the problem.
        if den.is_zero() {
            return PPoint::at_infinity(b_aff.0.sub(&a_aff.0)?, b_aff.1.sub(&a_aff.1)?);
        }
        let t_d = t.div(&den)?;
        Ok(PPoint::affine(
            a_aff.0.add(&b_aff.0.sub(&a_aff.0)?.mul(&t_d)?)?,
            a_aff.1.add(&b_aff.1.sub(&a_aff.1)?.mul(&t_d)?)?,
        ))
    }

    /// The line through a named scene segment, lifted to the projective
    /// completion. Exact: the segment's two points become homogeneous triples
    /// and the line is their cross product, so a Euclidean parallel pair
    /// becomes a pair of projective lines that meet at an ideal point.
    pub fn segment_line(graph: &SceneGraph, segment: &Segment) -> anyhow::Result<PLine> {
        let (x1, y1) = graph.coords(&segment.from)?;
        let (x2, y2) = graph.coords(&segment.to)?;
        PLine::through(&PPoint::affine(x1, y1), &PPoint::affine(x2, y2))
    }

    /// The exact [`Incidence`] verdict for two named scene segments in the
    /// projective plane: they always meet, and the answer says exactly where,
    /// including at infinity. This is the `AllLinesMeet` axiom as a decision
    /// procedure rather than a claim.
    pub fn incidence_of(
        graph: &SceneGraph,
        first: &Segment,
        second: &Segment,
    ) -> anyhow::Result<Incidence> {
        let (one, other) = (
            ProjectivePlane::segment_line(graph, first)?,
            ProjectivePlane::segment_line(graph, second)?,
        );
        let where_ = one.meet(&other)?;
        if where_.is_at_infinity() {
            Ok(Incidence::MeetAtInfinity(where_))
        } else {
            Ok(Incidence::Meet(where_))
        }
    }
}

/// Whether two homogeneous points are the same projective point. Two
/// coordinates' worth of proportionality is enough for two nonzero triples,
/// and all three are checked so the test is a decision rather than a hope.
fn same_point(p: &PPoint, q: &PPoint) -> anyhow::Result<bool> {
    let xy = p.x.mul(&q.y)?.sub(&p.y.mul(&q.x)?)?;
    let xz = p.x.mul(&q.z)?.sub(&p.z.mul(&q.x)?)?;
    let yz = p.y.mul(&q.z)?.sub(&p.z.mul(&q.y)?)?;
    Ok(xy.is_zero() && xz.is_zero() && yz.is_zero())
}

// ------------------------------------------------------- the affine plane ---

/// The affine plane: parallelism, exact ratios along a line, and no lengths.
///
/// The one thing this model has that the projective plane does not is the
/// distinction between "finite" and "at infinity" -- and that distinction is
/// carried by a *choice* of line at infinity rather than by anything intrinsic.
/// A parallel pair here is not two lines that never meet; it is two lines that
/// meet at one particular ideal point, and which ideal point they meet at is
/// a fact about the chart. So [`AffinePlane::parallel_lines_meet_at`] returns
/// the point, not a `bool`, and a reasoner that only gets a `bool` is one
/// projectivity away from answering differently.
///
/// And there are no lengths here. Not "lengths are hard" -- lengths are not
/// part of the structure: an affine map can stretch one direction without
/// limit while fixing the other, so the ratio of two segment lengths in
/// different directions is not a quantity of this plane. What survives is the
/// *ratio of two collinear segments*, which is what [`AffinePlane::ratio`]
/// computes, and which is exactly the affine invariant a textbook calls "the
/// ratio on a line".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AffinePlane;

impl AffinePlane {
    /// The affine plane's line at infinity, as a projective line.
    pub fn line_at_infinity() -> PLine {
        PLine::line_at_infinity()
    }

    /// The point at infinity of a direction `[dx : dy : 0]`, the representation
    /// of "where a parallel pair meets". Refusing a zero direction, because
    /// `[0 : 0 : 0]` is the one homogeneous triple that is not a point, and
    /// an implementation that returned it would be handing a reasoner a
    /// meeting point for every parallel pair at once.
    pub fn point_at_infinity(dx: Q, dy: Q) -> anyhow::Result<PPoint> {
        PPoint::at_infinity(dx, dy)
    }

    /// The exact [`Incidence`] verdict for two lines, in the affine chart.
    ///
    /// This is where the geometry states something a Euclidean reasoner cannot:
    /// a parallel pair *meets*, at an exact ideal point, and the verdict says
    /// so rather than returning "no meeting". The parallel test itself is the
    /// projective one, so the two models cannot disagree about which lines are
    /// parallel -- they differ only in whether they are asked about the meeting.
    pub fn incidence_of(
        graph: &SceneGraph,
        first: &Segment,
        second: &Segment,
    ) -> anyhow::Result<Incidence> {
        ProjectivePlane::incidence_of(graph, first, second)
    }

    /// The exact point at which two parallel lines meet in the completion, or
    /// an error naming the reason they are not parallel. A caller that wanted
    /// the boolean can derive it, but a caller that wanted to *use* the
    /// parallel axiom -- to fire a rule about the point at infinity, which is
    /// what the affine plane's theorems are about -- needs the point.
    pub fn parallel_lines_meet_at(
        graph: &SceneGraph,
        first: &Segment,
        second: &Segment,
    ) -> anyhow::Result<PPoint> {
        let (one, other) = (
            AffinePlane::line_of(graph, first)?,
            AffinePlane::line_of(graph, second)?,
        );
        let where_ = one.meet(&other)?;
        anyhow::ensure!(
            where_.is_at_infinity(),
            "{}",
            NonEuclideanError::OutOfScope {
                detail: format!(
                    "the lines meet at {}, which is a finite point: they are not a parallel pair, \
                     so there is no point at infinity to report",
                    where_.describe()
                ),
            }
        );
        Ok(where_)
    }

    /// The line through a named scene segment, as an affine line: the same
    /// projective line, understood as not being the line at infinity.
    pub fn line_of(graph: &SceneGraph, segment: &Segment) -> anyhow::Result<PLine> {
        ProjectivePlane::segment_line(graph, segment)
    }

    /// The exact directed ratio `AM / MB` of the three collinear points `a`,
    /// `m`, `b` -- the affine plane's one metric-like quantity, and the one
    /// that survives every affine map. A midpoint is the special case `1`.
    ///
    /// Refuses non-collinear input, coincident endpoints, and a middle point
    /// at an endpoint, all of which have no ratio: a segment is the unit this
    /// ratio is measured in, so a degenerate one has no value rather than the
    /// value infinity.
    pub fn ratio(a: &PPoint, m: &PPoint, b: &PPoint) -> anyhow::Result<Q> {
        let (a_aff, m_aff, b_aff) = (a.to_affine()?, m.to_affine()?, b.to_affine()?);
        // the parameter of m along a -> b, refusing a degenerate segment
        let (dx, dy) = (b_aff.0.sub(&a_aff.0)?, b_aff.1.sub(&a_aff.1)?);
        anyhow::ensure!(
            !dx.is_zero() || !dy.is_zero(),
            "{}",
            GeometryError::DegenerateSegment
        );
        let collinear = dx.mul(&m_aff.1.sub(&a_aff.1)?)?.sub(&dy.mul(&m_aff.0.sub(&a_aff.0)?)?)?;
        anyhow::ensure!(
            collinear.is_zero(),
            "{}",
            GeometryError::EmptyGeometry(format!(
                "the three points are not collinear, so the ratio along the line {}..{} is not \
                 defined",
                a.describe(),
                b.describe()
            ))
        );
        let t = if !dx.is_zero() {
            m_aff.0.sub(&a_aff.0)?.div(&dx)?
        } else {
            m_aff.1.sub(&a_aff.1)?.div(&dy)?
        };
        anyhow::ensure!(
            !t.is_zero() && !Q::ONE.less(&t),
            "{}",
            GeometryError::DegenerateSegment
        );
        // AM / MB = t / (1 - t)
        let one_minus = Q::ONE.sub(&t)?;
        anyhow::ensure!(
            !one_minus.is_zero(),
            "{}",
            GeometryError::DegenerateSegment
        );
        t.div(&one_minus)
    }

    /// The point dividing `ab` in the exact ratio `num : den`, the inverse of
    /// [`Self::ratio`]. A midpoint is `1 : 1`, and a trisection point is not
    /// a mystery in this geometry either.
    pub fn point_at_ratio(a: &PPoint, b: &PPoint, num: Q, den: Q) -> anyhow::Result<PPoint> {
        anyhow::ensure!(!num.is_zero() && !den.is_zero(), "{}", GeometryError::DegenerateSegment);
        let (a_aff, b_aff) = (a.to_affine()?, b.to_affine()?);
        let total = num.add(&den)?;
        let t = num.div(&total)?;
        Ok(PPoint::affine(
            a_aff.0.add(&b_aff.0.sub(&a_aff.0)?.mul(&t)?)?,
            a_aff.1.add(&b_aff.1.sub(&a_aff.1)?.mul(&t)?)?,
        ))
    }

    /// The segment length this plane does not have. The function exists to be
    /// refused, and its refusal says why: adding a length would mean adding a
    /// metric, which is leaving the affine plane for the Euclidean one. A
    /// caller that gets a number here has been handed a Euclidean assumption
    /// it did not declare.
    pub fn length_squared(&self, _a: &PPoint, _b: &PPoint) -> anyhow::Result<Q> {
        Err(NonEuclideanError::OutOfScope {
            detail: "the affine plane has no length: an affine map may stretch one direction \
                     without limit while fixing the other, so a squared distance is not one of \
                     its quantities. The affine invariant on a line is the ratio of collinear \
                     segments -- see AffinePlane::ratio."
                .to_string(),
        }
        .into())
    }
}

// ------------------------------------- the hyperbolic plane (Klein disk) ---

/// The hyperbolic plane in the Cayley-Klein (Klein) disk model: the unit disk
/// with its projective incidence, and the absolute conic `x^2 + y^2 = z^2` on
/// its boundary.
///
/// The model earns its keep on one point. In the Klein model, two lines whose
/// projective intersection lies *inside* the disk meet in the plane; whose
/// intersection lies *on* the boundary conic are asymptotically parallel (they
/// share an ideal point); and whose intersection lies *outside* the disk are
/// ultraparallel and share nothing at all. All three are exact rational
/// comparisons -- a sign of `x^2 + y^2 - z^2` -- so the hyperbolic axiom is
/// not asserted here, it is *decided*, and the reason Euclid's is false falls
/// out of the arithmetic: through a point off a line, any line aimed at a point
/// of that line outside the disk is a parallel, and there are as many such
/// points as there are rationals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CayleyKleinDisk {
    /// The squared radius of the disk: the absolute conic is
    /// `x^2 + y^2 = radius^2 z^2`, so the Euclidean unit disk is
    /// `radius_sq == 1` and a larger curvature `-1` disk is a smaller one.
    #[serde(with = "q_serde")]
    pub radius_sq: Q,
}

impl CayleyKleinDisk {
    /// The unit disk, the standard presentation.
    pub fn unit() -> Self {
        Self { radius_sq: Q::ONE }
    }

    /// A disk of the given squared radius, refusing a non-positive one: a disk
    /// of squared radius zero is a point, and "incidence in a point" is not a
    /// geometry this model can be.
    pub fn with_radius_sq(radius_sq: Q) -> anyhow::Result<Self> {
        anyhow::ensure!(
            Q::ZERO.less(&radius_sq),
            "{}",
            GeometryError::EmptyGeometry(format!(
                "a disk of squared radius {radius_sq} has no interior: the Klein model needs a \
                 positive radius to have incidence at all"
            ))
        );
        Ok(Self { radius_sq })
    }

    /// `x^2 + y^2 - radius^2 z^2` for a homogeneous point: the sign decides
    /// which of the three regions of the projective plane the point is in.
    pub fn conic_form(&self, point: &PPoint) -> anyhow::Result<Q> {
        point
            .x
            .mul(&point.x)?
            .add(&point.y.mul(&point.y)?)?
            .sub(&self.radius_sq.mul(&point.z)?.mul(&point.z)?)
    }

    /// Whether the point is in the interior of the disk -- the hyperbolic
    /// plane itself, as opposed to its ideal boundary or the outside.
    pub fn is_inside(&self, point: &PPoint) -> anyhow::Result<bool> {
        Ok(self.conic_form(point)?.less(&Q::ZERO))
    }

    /// Whether the point is on the absolute conic: an ideal point of the
    /// hyperbolic plane, where the two ends of a line live. This is the
    /// *only* place an "asymptotically parallel" pair meets, and it is a
    /// decision, not a limit.
    pub fn is_on_absolute(&self, point: &PPoint) -> anyhow::Result<bool> {
        Ok(self.conic_form(point)?.is_zero())
    }

    /// The exact [`Incidence`] verdict for two lines of the disk, which is the
    /// three-way split above. The `Disjoint` arm is the one Euclid does not
    /// have: it means the lines share no point at all, not even an ideal one,
    /// and reaching it is what "there is more than one parallel" means.
    pub fn incidence_of(&self, first: &PLine, second: &PLine) -> anyhow::Result<Incidence> {
        let where_ = first.meet(second)?;
        if self.is_inside(&where_)? {
            Ok(Incidence::Meet(where_))
        } else if self.is_on_absolute(&where_)? {
            Ok(Incidence::MeetAtInfinity(where_))
        } else {
            Ok(Incidence::Disjoint)
        }
    }

    /// The exact incidence of two named scene segments, in the disk model.
    pub fn incidence_of_segments(
        &self,
        graph: &SceneGraph,
        first: &Segment,
        second: &Segment,
    ) -> anyhow::Result<Incidence> {
        self.incidence_of(
            &ProjectivePlane::segment_line(graph, first)?,
            &ProjectivePlane::segment_line(graph, second)?,
        )
    }

    /// Exhibit exact lines through `point` that do not meet `line`.
    ///
    /// This is the hyperbolic axiom as a construction rather than a claim, and
    /// the construction is the whole argument: the line's own direction is a
    /// point at infinity, so adding a rational multiple of it to any point of
    /// the line gives another point of the line; those points leave the disk
    /// once the multiple is large enough; and the line from `point` to such a
    /// point cannot meet `line` inside the disk, because their only common
    /// point is that point. So one more than one such line is a witness that
    /// `UniqueParallelThroughPoint` fails here, and the witness is exact
    /// arithmetic all the way down.
    ///
    /// Refuses a point on the line, a line wholly outside the disk (nothing to
    /// be parallel to), and a point outside the disk. The bound on the search
    /// is refused rather than exceeded: a failure to find a witness within it
    /// is reported, not retried forever.
    pub fn non_merging_lines_through(
        &self,
        point: &PPoint,
        line: &PLine,
        wanted: usize,
    ) -> anyhow::Result<Vec<PLine>> {
        anyhow::ensure!(wanted > 0, "a witness exhibit must want at least one line");
        let (px, py) = point.to_affine()?;
        let direction = line.at_infinity()?;
        // Order matters, and the order is the one that names the *most*
        // specific failure. A point on the line is a better explanation for a
        // refusal than "that point is not in the disk", and a point on the
        // boundary of the disk is not a point of the plane at all -- which is
        // the honest reason to refuse it.
        anyhow::ensure!(
            !line.contains(point)?,
            "{}",
            GeometryError::DegenerateAngle {
                detail: "the point lies on the line, so no line through it misses that line: \
                         the parallels-through-a-point construction needs a point off the line"
                    .to_string(),
            }
        );
        anyhow::ensure!(
            self.is_inside(point)?,
            "{}",
            GeometryError::PointNotOnCircle {
                point: point.describe(),
                circle: "the Klein disk".to_string(),
            }
        );
        // A finite point of `line` to start from: the foot of the perpendicular
        // from the origin, which is the closest point of the line to the centre
        // of the disk and is rational whenever the line is. Any point of the
        // line would do; this one is exact and always finite, because
        // `a^2 + b^2` is nonzero for a line that has a direction at all.
        let norm_sq = line.a.mul(&line.a)?.add(&line.b.mul(&line.b)?)?;
        anyhow::ensure!(
            !norm_sq.is_zero(),
            "{}",
            GeometryError::DegenerateSegment
        );
        let base = PPoint::affine(
            line.a.mul(&line.c)?.neg()?.div(&norm_sq)?,
            line.b.mul(&line.c)?.neg()?.div(&norm_sq)?,
        );
        let mut found: Vec<PLine> = Vec::new();
        let mut step = Q::ONE;
        for _ in 0..16 {
            // base + step * direction, all homogeneous, so this is a point of
            // `line` by linearity and lands off the disk once step is big.
            let candidate = PPoint::new(
                base.x.add(&direction.x.mul(&step)?)?,
                base.y.add(&direction.y.mul(&step)?)?,
                base.z,
            )?;
            if !self.is_inside(&candidate)? {
                let through = PLine::through(&PPoint::affine(px, py), &candidate)?;
                // Verified, not asserted: the exhibit only counts a line the
                // exact incidence test agrees does not meet.
                if matches!(self.incidence_of(line, &through)?, Incidence::Disjoint) {
                    found.push(through);
                    if found.len() == wanted {
                        return Ok(found);
                    }
                }
            }
            step = step.mul(&Q::from_int(2))?;
        }
        anyhow::bail!(
            "{}",
            GeometryError::NoRationalSolution(format!(
                "no witness line through {} was found for this disk within the search bound; \
                 the construction is refusing rather than reporting an unverified exhibit",
                point.describe()
            ))
        )
    }
}

// -------------------------------------------- the sphere (great circles) ---

/// A ray of the sphere: an exact rational direction, with the two ends of the
/// ray identified.
///
/// A point of the sphere is a *ray*, not a direction with a sign, so `ray` and
/// its negative are one point -- which is why [`Self::antipodal`] is a
/// question with an exact answer and why [`Self::normalize`] divides by the
/// first nonzero coordinate, fixing a representative. Rational rays are dense
/// in the sphere, so this model is not a restriction to a finite set of
/// directions: any rational direction is available, and the incidence test is
/// a dot product.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SRay {
    #[serde(with = "q_serde")]
    pub x: Q,
    #[serde(with = "q_serde")]
    pub y: Q,
    #[serde(with = "q_serde")]
    pub z: Q,
}

impl SRay {
    /// A ray, refusing the zero triple: `(0,0,0)` is not a point of the sphere
    /// under any reading, and returning it would put a "point" on every great
    /// circle at once.
    pub fn new(x: Q, y: Q, z: Q) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !(x.is_zero() && y.is_zero() && z.is_zero()),
            "{}",
            GeometryError::EmptyGeometry("the triple (0, 0, 0) is not a direction, so it is not \
                 a point of the sphere: it lies on every great circle"
                .to_string())
        );
        Ok(Self { x, y, z })
    }

    /// The same ray with a fixed sign convention: the first nonzero coordinate
    /// made positive. Two rays that normalize alike are one point of the
    /// sphere, which is the antipodal identification made into a decision.
    pub fn normalize(&self) -> anyhow::Result<Self> {
        if !self.x.is_zero() {
            Ok(Self { x: Q::ONE, y: self.y.div(&self.x)?, z: self.z.div(&self.x)? })
        } else if !self.y.is_zero() {
            let positive = if self.y.less(&Q::ZERO) { self.y.neg()? } else { self.y };
            Ok(Self { x: Q::ZERO, y: Q::ONE, z: self.z.div(&positive)? })
        } else {
            anyhow::ensure!(!self.z.is_zero(), "{}", GeometryError::EmptyGeometry("the zero direction does not normalize".to_string()));
            // the sign is immaterial once the triple is (0, 0, +/-1), and the
            // normalize is the *identification* of antipodes, so the positive
            // representative is the one kept
            Ok(Self { x: Q::ZERO, y: Q::ZERO, z: Q::ONE })
        }
    }

    /// Whether two rays are antipodal, i.e. the same point of the sphere. This
    /// is the one predicate a planar kernel has no way to ask, and it is why
    /// spherical figures cannot be lifted by reusing the planar incidence code.
    pub fn antipodal(&self, other: &SRay) -> anyhow::Result<bool> {
        let a = self.normalize()?;
        let b = other.normalize()?;
        Ok(a.x == b.x && a.y == b.y && a.z == b.z)
    }

    /// The ray as a homogeneous point, for the shared [`Incidence`] verdict
    /// type. A ray *is* a point of the projective plane whose third coordinate
    /// is nonzero except at the two poles of no particular axis; the
    /// identification of `p` with `-p` is what makes it a sphere rather than a
    /// projective plane, and is carried by [`Self::antipodal`] rather than by
    /// the coordinates.
    pub fn to_point(&self) -> anyhow::Result<PPoint> {
        PPoint::new(self.x, self.y, self.z)
    }
}

impl fmt::Display for SRay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{} : {} : {}]", self.x, self.y, self.z)
    }
}

impl SRay {
    /// A one-line identity for reports, sign-normalized.
    pub fn describe(&self) -> String {
        match self.normalize() {
            Ok(ray) => ray.to_string(),
            Err(_) => self.to_string(),
        }
    }
}

/// The sphere with great-circle lines: rays as points, and a line as the set
/// of rays orthogonal to a fixed direction.
///
/// Incidence is a dot product and two lines meet in a cross product, so the
/// `AllLinesMeet` axiom of this geometry is a fact about exact rational
/// arithmetic rather than a hope: two distinct great circles have
/// non-proportional normals, and the cross product of two such normals is
/// never zero. That is the difference from the hyperbolic plane in one line of
/// algebra, and it is the reason the same `Incidence` type covers both: the
/// spherical verdict is always [`Incidence::Meet`], and the hyperbolic one
/// sometimes reaches [`Incidence::Disjoint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SphericalModel;

impl SphericalModel {
    /// The great circle perpendicular to a given direction, as its normal.
    /// A zero normal is refused: it is not a great circle, it is the empty
    /// set, and every ray would fail to lie on it.
    pub fn great_circle(normal: &SRay) -> anyhow::Result<PLine> {
        PLine::new(normal.x, normal.y, normal.z)
    }

    /// The exact [`Incidence`] verdict for two great circles. The only refusal
    /// is a pair of coincident circles, and the only `Ok` is a meeting -- which
    /// is what makes this the sharpest contrast with the hyperbolic model.
    pub fn incidence_of(&self, first: &PLine, second: &PLine) -> anyhow::Result<Incidence> {
        let where_ = first.meet(second)?;
        Ok(Incidence::Meet(where_))
    }

    /// The exact ray where two great circles meet, refusing a coincident
    /// pair. The cross product of the two normals is the direction of both
    /// meeting points, and is nonzero exactly when the circles are distinct --
    /// so "any two lines meet" is decided, not assumed.
    pub fn meet(&self, first: &PLine, second: &PLine) -> anyhow::Result<SRay> {
        let where_ = first.meet(second)?;
        SRay::new(where_.x, where_.y, where_.z)
    }

    /// The exact direction where the two great circles determined by two
    /// segments of a planar scene meet, used as the witness in an axiom report
    /// for a spherical reading of a figure. The planar points are lifted to
    /// rays by appending a `1`, so a planar segment becomes a great circle
    /// through the origin, and the two such circles meet in the cross product
    /// of their normals.
    pub fn meet_of_segments(
        graph: &SceneGraph,
        first: &Segment,
        second: &Segment,
    ) -> anyhow::Result<SRay> {
        let (first_normal, second_normal) =
            (Self::circle_of(graph, first)?, Self::circle_of(graph, second)?);
        SphericalModel.meet(&first_normal, &second_normal)
    }

    /// The great circle through two named scene points, as its normal.
    pub fn normal_through(graph: &SceneGraph, a: &str, b: &str) -> anyhow::Result<PLine> {
        let one = Self::ray_at(graph, a)?;
        let other = Self::ray_at(graph, b)?;
        SphericalModel.normal_of(&one, &other)
    }

    /// The normal of the great circle through two rays: the cross product,
    /// refusing a pair of antipodal rays. Two antipodal points do not
    /// determine a line on the sphere -- every great circle through the axis
    /// contains both -- and a model that picked one silently would be
    /// reporting a figure the problem never described.
    pub fn normal_of(&self, first: &SRay, second: &SRay) -> anyhow::Result<PLine> {
        anyhow::ensure!(
            !first.antipodal(second)?,
            "{}",
            GeometryError::DegenerateSegment
        );
        PLine::new(
            first.y.mul(&second.z)?.sub(&first.z.mul(&second.y)?)?,
            first.z.mul(&second.x)?.sub(&first.x.mul(&second.z)?)?,
            first.x.mul(&second.y)?.sub(&first.y.mul(&second.x)?)?,
        )
    }

    /// The ray lifting a named scene point: `(x, y, 1)`, a direction with a
    /// nonzero third coordinate, so every planar point is a genuine direction.
    pub fn ray_at(graph: &SceneGraph, name: &str) -> anyhow::Result<SRay> {
        let (x, y) = graph.coords(name)?;
        SRay::new(x, y, Q::ONE)
    }

    /// The great circle through the two endpoints of a named segment.
    pub fn circle_of(graph: &SceneGraph, segment: &Segment) -> anyhow::Result<PLine> {
        Self::normal_through(graph, &segment.from, &segment.to)
    }
}

// -------------------------------------------------------------------- tests --

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
    use crate::geomkernel::{graph_from_points, KPoint};

    /// An exact rational, for the values a test asserts against.
    fn q(num: i128, den: i128) -> Q {
        Q::new(num, den).unwrap()
    }

    /// A lattice point, which is what the planar figures below are made of.
    fn kp(name: &str, x: i64, y: i64) -> KPoint {
        KPoint { name: name.to_string(), x: Frac::from_int(x), y: Frac::from_int(y) }
    }

    /// A segment between two of those names.
    fn seg(from: &str, to: &str) -> Segment {
        Segment { from: from.to_string(), to: to.to_string() }
    }

    /// A scene in a declared geometry, built from points and given statements.
    fn scene(geometry: &str, points: Vec<KPoint>, facts: Vec<Constraint>) -> SceneGraph {
        let mut graph = graph_from_points(points, facts);
        graph.geometry = geometry.to_string();
        graph
    }

    /// A unit-disk Klein model.
    fn klein() -> CayleyKleinDisk {
        CayleyKleinDisk::unit()
    }

    /// The refusal a call produced, read as the typed error this module says
    /// it produces. A refusal that arrives as a string would mean the type is
    /// not doing the work the prose claims.
    fn refusal<T>(result: anyhow::Result<T>) -> NonEuclideanError {
        result
            .err()
            .and_then(|why| why.downcast::<NonEuclideanError>().ok())
            .expect("a typed NonEuclideanError")
    }

    /// The message a refusal produced, whichever typed error carried it. Used
    /// where the refusal is the *kernel's* -- a degenerate segment, a zero
    /// radius -- since those are the kernel's vocabulary and this module
    /// deliberately reuses it rather than growing a second hierarchy.
    fn message<T>(result: anyhow::Result<T>) -> String {
        match result.err() {
            Some(why) => {
                // the typed error's own Display, which is what a caller reads
                let text = why.to_string();
                assert!(!text.is_empty(), "a refusal with no message");
                text
            }
            None => panic!("expected a refusal, got Ok"),
        }
    }

    // ---------------------------------------------------- the environment --

    #[test]
    fn test_every_geometry_kind_round_trips_through_serde() {
        for kind in GeometryKind::all() {
            let json = serde_json::to_string(kind).unwrap();
            let back: GeometryKind = serde_json::from_str(&json).unwrap();
            assert_eq!(back, *kind, "{kind} did not survive a serde round trip");
            // and the wire name is the name the scene carries
            assert!(
                json.contains(&format!("\"{}\"", kind.as_str())),
                "{json} does not name {kind}"
            );
        }
    }

    #[test]
    fn test_every_axiom_round_trips_through_serde() {
        for axiom in Axiom::all() {
            let json = serde_json::to_string(axiom).unwrap();
            let back: Axiom = serde_json::from_str(&json).unwrap();
            assert_eq!(back, *axiom);
            assert!(
                json.contains(&format!("\"{}\"", axiom.as_str())),
                "{json} does not carry the wire name of {axiom}"
            );
        }
    }

    #[test]
    fn test_geometry_names_parse_and_unknown_names_are_refused() {
        assert_eq!("euclidean".parse::<GeometryKind>().unwrap(), GeometryKind::Euclidean);
        assert_eq!("  Projective ".parse::<GeometryKind>().unwrap(), GeometryKind::Projective);
        // the classical synonyms are accepted, because the literature uses them
        assert_eq!("elliptic".parse::<GeometryKind>().unwrap(), GeometryKind::Spherical);
        assert_eq!("Boltzmann".parse::<GeometryKind>().unwrap(), GeometryKind::Hyperbolic);
        // and anything else is an error carrying the valid list, not a default
        let why = refusal("hyperbolic-ish".parse::<GeometryKind>());
        assert_eq!(why, NonEuclideanError::UnknownGeometry { name: "hyperbolic-ish".to_string() });
        let text = why.to_string();
        for name in GeometryKind::names() {
            assert!(text.contains(name), "the refusal does not offer {name}: {text}");
        }
    }

    #[test]
    fn test_the_parallel_axiom_is_true_in_four_of_five_geometries() {
        // The audit's sharpest case, as a table: the Euclidean parallel axiom
        // holds Euclideanly and affinely and fails in the other three.
        assert!(GeometryKind::Euclidean.asserts(Axiom::UniqueParallelThroughPoint));
        assert!(GeometryKind::Affine.asserts(Axiom::UniqueParallelThroughPoint));
        assert!(!GeometryKind::Projective.asserts(Axiom::UniqueParallelThroughPoint));
        assert!(!GeometryKind::Spherical.asserts(Axiom::UniqueParallelThroughPoint));
        assert!(!GeometryKind::Hyperbolic.asserts(Axiom::UniqueParallelThroughPoint));
    }

    #[test]
    fn test_unique_parallel_is_false_in_spherical_and_true_only_in_two() {
        assert!(!GeometryKind::Spherical.asserts(Axiom::UniqueParallelThroughPoint));
        let holders = Axiom::UniqueParallelThroughPoint.held_by();
        assert_eq!(holders, vec![GeometryKind::Euclidean, GeometryKind::Affine]);
        // and the spherical reason is named rather than merely absent
        assert!(GeometryKind::Spherical.asserts(Axiom::NoUniqueParallel));
        assert!(GeometryKind::Spherical.asserts(Axiom::AllLinesMeet));
    }

    #[test]
    fn test_the_angle_sum_axiom_differs_per_geometry() {
        assert_eq!(GeometryKind::Euclidean.angle_sum_axiom(), Some(Axiom::AnglesSumTo180));
        assert_eq!(GeometryKind::Spherical.angle_sum_axiom(), Some(Axiom::AnglesSumTo360));
        assert_eq!(GeometryKind::Hyperbolic.angle_sum_axiom(), Some(Axiom::AnglesSumBelow180));
        // the two non-metric geometries have no angle axiom at all, and saying
        // so is better than defaulting them to 180
        assert_eq!(GeometryKind::Affine.angle_sum_axiom(), None);
        assert_eq!(GeometryKind::Projective.angle_sum_axiom(), None);
        // and the three that differ are genuinely different claims
        assert!(GeometryKind::Euclidean.asserts(Axiom::AnglesSumTo180));
        assert!(!GeometryKind::Hyperbolic.asserts(Axiom::AnglesSumTo180));
        assert!(!GeometryKind::Spherical.asserts(Axiom::AnglesSumTo180));
    }

    #[test]
    fn test_no_geometry_asserts_the_whole_table() {
        // Every axiom must be held by at least one geometry, and no geometry by
        // all of them: an environment that asserted everything would license
        // every rule, which is the over-application in a different costume.
        for axiom in Axiom::all() {
            let holders = axiom.held_by();
            assert!(!holders.is_empty(), "no geometry holds {axiom}: {axiom}");
            // the incidence axiom is the deliberate exception, and it is the
            // point of the table: it is the *only* claim all five share, which
            // is exactly why sharing it is not enough
            if *axiom != Axiom::IncidenceAxiom {
                assert!(holders.len() < GeometryKind::ALL.len(), "{axiom} is held everywhere");
            }
        }
        assert_eq!(Axiom::IncidenceAxiom.held_by().len(), GeometryKind::ALL.len());
        for kind in GeometryKind::all() {
            assert!(kind.axioms().len() < Axiom::ALL.len(), "{kind} asserts everything");
            // and the incidence axiom is the one thing all five share
            assert!(kind.asserts(Axiom::IncidenceAxiom), "{kind} lacks even incidence");
        }
    }

    #[test]
    fn test_every_axiom_name_is_unique_and_round_trips() {
        let mut names: Vec<&str> = Axiom::all().iter().map(|axiom| axiom.as_str()).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "two axioms share a wire name");
    }

    // ------------------------------------------------ the permitted theorems --

    #[test]
    fn test_a_theorem_the_environment_does_not_permit_is_refused() {
        // The engine's question, asked of a geometry that does not license it.
        assert!(!GeometryKind::Projective.permits("triangle_angle_sum"));
        let why = refusal(GeometryKind::Projective.require_theorem("triangle_angle_sum"));
        match &why {
            NonEuclideanError::TheoremNotPermitted { geometry, theorem, permitted } => {
                assert_eq!(*geometry, GeometryKind::Projective);
                assert_eq!(theorem, "triangle_angle_sum");
                // the refusal says what *would* be licensed, so a caller can
                // recover rather than merely stop
                assert!(permitted.contains(&"cross_ratio_invariant".to_string()));
                assert!(!permitted.contains(theorem));
            }
            other => panic!("expected a theorem refusal, got {other:?}"),
        }
        assert!(!GeometryKind::Hyperbolic.permits("pythagoras"));
        assert!(!GeometryKind::Spherical.permits("unique_parallel_through_point"));
    }

    #[test]
    fn test_a_permitted_theorem_is_licensed_by_name() {
        for kind in GeometryKind::all() {
            for theorem in kind.valid_theorems() {
                assert!(kind.permits(theorem), "{kind} lists {theorem} but does not permit it");
                assert_eq!(kind.require_theorem(theorem).unwrap(), *theorem);
            }
        }
    }

    #[test]
    fn test_the_kernel_rule_names_are_licensed_only_where_they_hold() {
        // The kernel's own rule library was written under Euclid, so the
        // Euclidean environment licenses it and the projective one does not
        // license a theorem whose proof needs the parallel postulate.
        assert!(GeometryKind::Euclidean.permits("MidpointCollinear"));
        assert!(GeometryKind::Euclidean.permits("pythagoras"));
        assert!(!GeometryKind::Projective.permits("pythagoras"));
        assert!(!GeometryKind::Projective.permits("MidpointBisectsPerpendicular"));
        // a rule name that is a near-miss of a licensed one is not licensed
        assert!(!GeometryKind::Euclidean.permits("MidpointCollinear "));
        assert!(!GeometryKind::Euclidean.permits("midpointcollinear"));
    }

    #[test]
    fn test_authorize_reads_the_scene_not_the_caller() {
        let euclidean = scene("euclidean", vec![kp("A", 0, 0)], vec![]);
        assert_eq!(authorize(&euclidean, "pythagoras").unwrap(), GeometryKind::Euclidean);
        // the same rule under a different declaration is refused, which is the
        // whole point of declaring the geometry on the scene
        let projective = scene("projective", vec![kp("A", 0, 0)], vec![]);
        let why = refusal(authorize(&projective, "pythagoras"));
        assert!(matches!(why, NonEuclideanError::TheoremNotPermitted { .. }), "{why:?}");
    }

    // ------------------------------------------------ the projective plane --

    #[test]
    fn test_projective_cross_ratio_is_exact_on_a_known_configuration() {
        // Four lattice points on the x-axis, parameterized so that A = 0 and
        // B = 1: C at 2, D at 3. Then
        //   (A,B;C,D) = (c/d) * ((d-1)/(c-1)) = (2/3) * (2/1) = 4/3.
        // Hand-computed, and a float would have to be trusted to produce it.
        let (a, b, c, d) = (
            PPoint::affine(Q::ZERO, Q::ZERO),
            PPoint::affine(Q::ONE, Q::ZERO),
            PPoint::affine(Q::from_int(2), Q::ZERO),
            PPoint::affine(Q::from_int(3), Q::ZERO),
        );
        assert_eq!(ProjectivePlane::cross_ratio(&a, &b, &c, &d).unwrap(), q(4, 3));
        // (C, D; A, B) is the *same* value -- a cross ratio is invariant under
        // that pairing -- and swapping C with D inverts it. Both exactly, and
        // the second is how a 4/3 is told apart from a 3/4.
        assert_eq!(ProjectivePlane::cross_ratio(&c, &d, &a, &b).unwrap(), q(4, 3));
        assert_eq!(ProjectivePlane::cross_ratio(&a, &b, &d, &c).unwrap(), q(3, 4));
    }

    #[test]
    fn test_projective_cross_ratio_of_the_harmonic_quadruple_is_minus_one() {
        // (0, 1, 1/2, infinity) is the harmonic quadruple: the fourth point is
        // the point at infinity of the line, so (A,B;C,D) = (1/2) * (1/(-1/2)) = -1.
        // This is the test that pins the convention down -- the other common
        // conventions differ by an inversion, and -1 is where they part.
        let (a, b, c) = (
            PPoint::affine(Q::ZERO, Q::ZERO),
            PPoint::affine(Q::ONE, Q::ZERO),
            PPoint::affine(Q::new(1, 2).unwrap(), Q::ZERO),
        );
        let infinity = PPoint::at_infinity(Q::ONE, Q::ZERO).unwrap();
        assert!(infinity.is_at_infinity());
        // The ideal point is decided by the same determinant form as any other
        // quadruple, and gives exactly -1. This is the case a reasoner reading
        // an affine parameter would refuse, because the parameter of D is
        // infinite: refusing it would refuse the harmonic range, which is the
        // one configuration projective geometry is for.
        assert_eq!(ProjectivePlane::cross_ratio(&a, &b, &c, &infinity).unwrap(), q(-1, 1));
        // And the construction agrees: the harmonic conjugate of the midpoint
        // of AB is the point at *infinity* of the line -- returned as a point,
        // not refused, because "the midpoint's conjugate is at infinity" is
        // precisely what the projective plane is for. Its cross ratio against
        // the same three points is the same -1.
        let d = ProjectivePlane::harmonic_conjugate(&a, &b, &c).unwrap();
        assert!(d.is_at_infinity());
        assert_eq!(d.describe(), "[1 : 0 : 0]");
        assert_eq!(ProjectivePlane::cross_ratio(&a, &b, &c, &d).unwrap(), q(-1, 1));
        // A C that is not the midpoint has a *finite* conjugate, and the value
        // is still exactly -1: take C at 1/3, whose conjugate is at 1/2.
        let third = PPoint::affine(Q::new(1, 3).unwrap(), Q::ZERO);
        let conjugate = ProjectivePlane::harmonic_conjugate(&a, &b, &third).unwrap();
        assert!(!conjugate.is_at_infinity());
        assert_eq!(conjugate.to_affine().unwrap().0, q(1, 2));
        assert_eq!(ProjectivePlane::cross_ratio(&a, &b, &third, &conjugate).unwrap(), q(-1, 1));
    }

    #[test]
    fn test_projective_line_intersection_is_exact() {
        // y = 0 and x = 0 meet at the origin, exactly, in homogeneous form
        let (horizontal, vertical) = (
            PLine::through(&PPoint::affine(Q::ZERO, Q::ZERO), &PPoint::affine(Q::ONE, Q::ZERO)).unwrap(),
            PLine::through(&PPoint::affine(Q::ZERO, Q::ZERO), &PPoint::affine(Q::ZERO, Q::ONE)).unwrap(),
        );
        let where_ = horizontal.meet(&vertical).unwrap();
        assert_eq!(where_.to_affine().unwrap(), (Q::ZERO, Q::ZERO));
        // 3y = 4x and 2x = y meet at the origin too, with different coefficients
        let (first, second) = (
            PLine::new(q(-4, 1), q(3, 1), Q::ZERO).unwrap(),
            PLine::new(q(2, 1), q(-1, 1), Q::ZERO).unwrap(),
        );
        assert_eq!(first.meet(&second).unwrap().to_affine().unwrap(), (Q::ZERO, Q::ZERO));
    }

    #[test]
    fn test_every_two_projective_lines_meet_and_coincident_ones_refuse() {
        // Two distinct lines: a meeting, always. (AllLinesMeet as code.)
        let (one, other) = (
            PLine::new(Q::ONE, Q::ZERO, q(-1, 1)).unwrap(),
            PLine::new(Q::ZERO, Q::ONE, q(-1, 1)).unwrap(),
        );
        let where_ = one.meet(&other).unwrap();
        assert_eq!(where_.to_affine().unwrap(), (Q::ONE, Q::ONE));
        // the same line twice is not a meeting: it is every point
        let err = one.meet(&one).unwrap_err();
        assert!(err.to_string().contains("coincident"), "{err}");
    }

    #[test]
    fn test_parallel_lines_meet_at_an_exact_point_at_infinity() {
        // y = 0 and y = 1 are parallel in the affine chart and meet at [1:0:0]
        // in the projective plane. The direction of y = 0 is (1, 0).
        let (first, second) = (
            PLine::new(Q::ZERO, Q::ONE, Q::ZERO).unwrap(),
            PLine::new(Q::ZERO, Q::ONE, q(-1, 1)).unwrap(),
        );
        let where_ = first.meet(&second).unwrap();
        assert!(where_.is_at_infinity());
        assert_eq!(where_.to_affine().unwrap_err().to_string().contains("point at infinity"), true);
        assert_eq!(where_.describe(), "[1 : 0 : 0]");
        assert!(first.meets_at_infinity_of(&second).unwrap());
        // and a line that is not parallel does not share an ideal point
        let (third, fourth) = (
            PLine::new(Q::ZERO, Q::ONE, Q::ZERO).unwrap(),
            PLine::new(Q::ONE, Q::ZERO, Q::ZERO).unwrap(),
        );
        assert!(!third.meets_at_infinity_of(&fourth).unwrap());
    }

    #[test]
    fn test_the_zero_triple_is_not_a_point_and_not_a_line() {
        // (0,0,0) is the one homogeneous triple that is neither: not a point
        // at infinity, and not a line either.
        let err = PPoint::new(Q::ZERO, Q::ZERO, Q::ZERO).unwrap_err();
        assert!(err.to_string().contains("(0, 0, 0)"), "{err}");
        let err = PLine::new(Q::ZERO, Q::ZERO, Q::ZERO).unwrap_err();
        assert!(err.to_string().contains("not a line"), "{err}");
        // and a direction of (0,0) has no point at infinity, because that would
        // be the same non-point
        let err = AffinePlane::point_at_infinity(Q::ZERO, Q::ZERO).unwrap_err();
        assert!(err.to_string().contains("(0, 0, 0)"), "{err}");
    }

    // --------------------------------------------------- the affine plane --

    #[test]
    fn test_the_affine_point_at_infinity_of_a_direction_is_exact() {
        // Direction (2, 3) is the point [2 : 3 : 0], and scaling does not make
        // it a different point: [4 : 6 : 0] is the same one.
        let (one, scaled) = (
            AffinePlane::point_at_infinity(Q::from_int(2), Q::from_int(3)).unwrap(),
            AffinePlane::point_at_infinity(Q::from_int(4), Q::from_int(6)).unwrap(),
        );
        assert!(one.is_at_infinity() && scaled.is_at_infinity());
        assert!(same_point(&one, &scaled).unwrap());
        assert_eq!(one.describe(), scaled.describe());
        assert!(!one.to_affine().is_ok(), "an ideal point has no affine coordinates");
    }

    #[test]
    fn test_affine_parallel_lines_meet_at_their_shared_point_at_infinity() {
        // Two horizontal segments at different heights: parallel in the affine
        // plane, and the plane says exactly where they meet.
        let graph = scene(
            "affine",
            vec![kp("A", 0, 0), kp("B", 2, 0), kp("C", 0, 1), kp("D", 2, 1)],
            vec![],
        );
        let ideal = AffinePlane::parallel_lines_meet_at(&graph, &seg("A", "B"), &seg("C", "D")).unwrap();
        assert!(ideal.is_at_infinity());
        assert_eq!(ideal.describe(), "[1 : 0 : 0]");
        // and the incidence verdict says "meets at infinity", not "disjoint":
        // in the affine plane a parallel pair is a meeting
        let verdict = AffinePlane::incidence_of(&graph, &seg("A", "B"), &seg("C", "D")).unwrap();
        assert!(!verdict.meets(), "the two horizontal lines do not meet inside the plane");
        assert!(matches!(verdict, Incidence::MeetAtInfinity(_)));
    }

    #[test]
    fn test_affine_ratios_along_a_line_are_exact() {
        let (a, b) = (PPoint::affine(Q::ZERO, Q::ZERO), PPoint::affine(Q::from_int(4), Q::ZERO));
        // the midpoint is the 1:1 case, and the ratio is exactly 1
        let mid = AffinePlane::point_at_ratio(&a, &b, Q::ONE, Q::ONE).unwrap();
        assert_eq!(mid.to_affine().unwrap(), (Q::from_int(2), Q::ZERO));
        assert_eq!(AffinePlane::ratio(&a, &mid, &b).unwrap(), q(1, 1));
        // a quarter of the way along is 1:3, exactly
        let quarter = AffinePlane::point_at_ratio(&a, &b, Q::ONE, Q::from_int(3)).unwrap();
        assert_eq!(AffinePlane::ratio(&a, &quarter, &b).unwrap(), q(1, 3));
        assert_eq!(quarter.to_affine().unwrap().0, Q::ONE);
        // and the ratio of a non-collinear triple is refused rather than
        // approximated: a ratio is a statement about a line
        let off = PPoint::affine(Q::ONE, Q::ONE);
        assert!(AffinePlane::ratio(&a, &off, &b).is_err());
    }

    #[test]
    fn test_the_affine_plane_refuses_to_give_a_length() {
        // The refusal is the point: an affine map can stretch one direction
        // without limit, so a squared distance is not a quantity of this plane.
        let (a, b) = (PPoint::affine(Q::ZERO, Q::ZERO), PPoint::affine(Q::ONE, Q::ZERO));
        let why = refusal(AffinePlane.length_squared(&a, &b));
        match why {
            NonEuclideanError::OutOfScope { detail } => {
                assert!(detail.contains("no length"), "{detail}");
                assert!(detail.contains("ratio"), "{detail}");
            }
            other => panic!("expected an out-of-scope refusal, got {other:?}"),
        }
    }

    // ---------------------------------------- the hyperbolic plane (Klein) --

    #[test]
    fn test_through_a_point_off_a_line_pass_several_lines_that_do_not_meet_it() {
        // The exhibit. The x-axis is a chord of the disk; P is off it, inside.
        // Two exact lines through P are built and each is *checked* by the
        // exact incidence test to share no point with the x-axis, not even an
        // ideal one. In the Euclidean plane exactly one such line exists.
        let disk = klein();
        let axis = PLine::new(Q::ZERO, Q::ONE, Q::ZERO).unwrap();
        let off = PPoint::affine(Q::ZERO, Q::new(1, 2).unwrap());
        assert!(disk.is_inside(&off).unwrap(), "P must be inside the disk");
        assert!(!axis.contains(&off).unwrap(), "P must be off the line");

        let witnesses = disk.non_merging_lines_through(&off, &axis, 2).unwrap();
        assert_eq!(witnesses.len(), 2, "the axiom needs at least two non-meeting lines");
        assert_ne!(witnesses[0].describe(), witnesses[1].describe(), "the two witnesses are the same line");
        for witness in &witnesses {
            // the exact test, not a picture
            assert!(
                matches!(disk.incidence_of(&axis, witness).unwrap(), Incidence::Disjoint),
                "{witness} should not meet the axis at all, but the exact test says it does"
            );
            // and each witness does pass through P
            assert!(witness.contains(&off).unwrap());
        }
        // so `UniqueParallelThroughPoint` has two counterexamples here, and the
        // hyperbolic axiom is the one that is true
        assert!(!GeometryKind::Hyperbolic.asserts(Axiom::UniqueParallelThroughPoint));
        assert!(GeometryKind::Hyperbolic.asserts(Axiom::InfinitelyManyParallelsThroughPoint));
    }

    #[test]
    fn test_a_klein_line_that_is_a_chord_meets_another_inside_the_disk() {
        // The negative control for the exhibit: two chords of the disk really
        // do meet, inside it, so `Disjoint` is a decision and not a default.
        let disk = klein();
        let (first, second) = (
            PLine::new(Q::ZERO, Q::ONE, Q::ZERO).unwrap(),
            PLine::new(Q::ONE, Q::ZERO, Q::ZERO).unwrap(),
        );
        let where_ = disk.incidence_of(&first, &second).unwrap();
        assert!(where_.meets());
        assert_eq!(where_.at().unwrap().to_affine().unwrap(), (Q::ZERO, Q::ZERO));
    }

    #[test]
    fn test_a_klein_asymptotically_parallel_pair_meets_on_the_absolute() {
        // The hyperbolic "parallel" of the first kind: two lines sharing an
        // ideal point, which is a point of the absolute conic. The pair is
        // built by hand -- the x-axis chord, and a line through the conic point
        // [1 : 0 : 1] and the interior point [0 : 1 : 2] -- so the meeting is a
        // design decision rather than a hope.
        let disk = klein();
        let through_centre = PLine::new(Q::ZERO, Q::ONE, Q::ZERO).unwrap();
        let parallel = PLine::new(Q::ONE, Q::from_int(2), q(-1, 1)).unwrap();
        let verdict = disk.incidence_of(&through_centre, &parallel).unwrap();
        assert!(matches!(verdict, Incidence::MeetAtInfinity(_)), "{verdict}");
        let ideal = verdict.at().unwrap();
        // the meeting is [1 : 0 : 1], and it is on the absolute exactly:
        // 1^2 + 0^2 - 1^2 = 0, a sign test on exact rationals
        assert!(disk.is_on_absolute(ideal).unwrap(), "the meeting must be on the absolute");
        assert_eq!(ideal.describe(), "[1 : 0 : 1]");
        // it is a *finite* point of the conic -- the ideal points of the
        // hyperbolic plane are boundary points, not points at projective
        // infinity, which is exactly why they are invisible to a chart
        assert_eq!(ideal.to_affine().unwrap(), (Q::ONE, Q::ZERO));
    }

    #[test]
    fn test_klein_chords_that_are_euclideanly_parallel_are_ultraparallel() {
        // The two kinds of non-meeting, told apart by an exact test -- and this
        // is the case a Euclidean reasoner gets wrong in the other direction.
        // Two chords of the disk that are parallel as Euclidean lines share no
        // ideal point, because the projective intersection is [1 : 0 : 0],
        // which is *outside* the conic (1 - 0 = 1 > 0). So the Euclidean
        // "exactly one parallel through a point" is an ultraparallel statement
        // here, not an asymptotic one, and the model says so by arithmetic.
        let disk = klein();
        let (bottom, top) = (
            PLine::new(Q::ZERO, Q::ONE, Q::ZERO).unwrap(),
            PLine::new(Q::ZERO, Q::ONE, q(-1, 2)).unwrap(),
        );
        let where_ = bottom.meet(&top).unwrap();
        assert_eq!(where_.describe(), "[1 : 0 : 0]");
        assert!(!disk.is_on_absolute(&where_).unwrap());
        assert!(matches!(disk.incidence_of(&bottom, &top).unwrap(), Incidence::Disjoint));
        // while a pair through a common conic point does meet at one
        let asymptotic = PLine::new(Q::ONE, Q::from_int(2), q(-1, 1)).unwrap();
        assert!(matches!(disk.incidence_of(&bottom, &asymptotic).unwrap(), Incidence::MeetAtInfinity(_)));
    }

    #[test]
    fn test_the_klein_disk_refuses_degenerate_requests() {
        let disk = klein();
        let axis = PLine::new(Q::ZERO, Q::ONE, Q::ZERO).unwrap();
        let off = PPoint::affine(Q::ZERO, Q::new(1, 2).unwrap());
        // a point on the line has no lines through it missing that line. The
        // point is inside the disk as well as on the line, so the refusal is
        // about the parallelism and not about the point.
        let on_line = PPoint::affine(Q::new(1, 2).unwrap(), Q::ZERO);
        assert!(axis.contains(&on_line).unwrap());
        assert!(disk.is_inside(&on_line).unwrap());
        let why = message(disk.non_merging_lines_through(&on_line, &axis, 1));
        assert!(why.contains("lies on the line"), "{why}");
        // a point outside the disk is not a point of the hyperbolic plane
        let outside = PPoint::affine(Q::from_int(5), Q::ZERO);
        assert!(!disk.is_inside(&outside).unwrap());
        let why = message(disk.non_merging_lines_through(&outside, &axis, 1));
        assert!(why.contains("Klein disk"), "{why}");
        // and a disk with no interior is refused rather than accepted
        let why = message(CayleyKleinDisk::with_radius_sq(Q::ZERO));
        assert!(why.contains("no interior"), "{why}");
        // a witness exhibit of zero lines is a contradiction in terms
        assert!(disk.non_merging_lines_through(&off, &axis, 0).is_err());
    }

    #[test]
    fn test_the_absolute_conic_is_decided_exactly() {
        // (1, 0, 1) is on the unit disk's absolute: 1 + 0 - 1 = 0. A float
        // test of a conic is a bug waiting for a near-miss; this is a sign.
        let disk = klein();
        assert!(disk.is_on_absolute(&PPoint::affine(Q::ONE, Q::ZERO)).unwrap());
        assert!(disk.is_inside(&PPoint::affine(Q::ZERO, Q::ZERO)).unwrap());
        // (1, 1) is *outside* the unit disk: 1 + 1 - 1 = 1, positive. The
        // negative arm is a point of the disk, and the test is the sign.
        assert!(!disk.is_inside(&PPoint::affine(Q::ONE, Q::ONE)).unwrap());
        // the conic form is the exact quantity the three-way split reads
        assert_eq!(disk.conic_form(&PPoint::affine(Q::ONE, Q::ZERO)).unwrap(), Q::ZERO);
        assert!(disk.conic_form(&PPoint::affine(Q::ZERO, Q::ZERO)).unwrap().less(&Q::ZERO));
        assert!(!disk.conic_form(&PPoint::affine(Q::ONE, Q::ONE)).unwrap().less(&Q::ZERO));
        // and (2, 0) is far outside, so a "point" of the disk model may in fact
        // be no point of the hyperbolic plane at all
        assert!(!disk.is_inside(&PPoint::affine(Q::from_int(2), Q::ZERO)).unwrap());
        assert!(!disk.is_on_absolute(&PPoint::affine(Q::from_int(2), Q::ZERO)).unwrap());
    }

    // ------------------------------------------------ the spherical model --

    #[test]
    fn test_two_great_circles_meet_in_an_exact_direction() {
        // The exhibit for `AllLinesMeet` on the sphere: the equator (normal
        // (0,0,1)) and the prime meridian through the poles (normal (1,0,0))
        // meet in the direction of their cross product, (0, 1, 0) up to sign --
        // exactly, with no trigonometry anywhere.
        let model = SphericalModel;
        let (equator, meridian) = (
            PLine::new(Q::ZERO, Q::ZERO, Q::ONE).unwrap(),
            PLine::new(Q::ONE, Q::ZERO, Q::ZERO).unwrap(),
        );
        let where_ = model.meet(&equator, &meridian).unwrap();
        assert_eq!(where_.describe(), "[0 : 1 : 0]");
        assert!(model.incidence_of(&equator, &meridian).unwrap().meets());
        // the meeting ray lies on both circles, checked by dot product
        let point = where_.to_point().unwrap();
        assert!(equator.contains(&point).unwrap());
        assert!(meridian.contains(&point).unwrap());
        // and the antipodal ray is the same point of the sphere
        let antipode = SRay::new(Q::ZERO, q(-1, 1), Q::ZERO).unwrap();
        assert!(where_.antipodal(&antipode).unwrap());
    }

    #[test]
    fn test_every_two_distinct_great_circles_meet() {
        // Three unrelated great circles, all meeting, each time exactly. The
        // only refusal in this model is a coincident pair.
        let model = SphericalModel;
        let circles = [
            PLine::new(Q::ONE, Q::from_int(2), Q::from_int(3)).unwrap(),
            PLine::new(q(-2, 1), Q::ONE, Q::from_int(4)).unwrap(),
            PLine::new(Q::from_int(5), q(-3, 1), Q::from_int(7)).unwrap(),
        ];
        for (i, one) in circles.iter().enumerate() {
            for (j, other) in circles.iter().enumerate() {
                if i == j {
                    continue;
                }
                let verdict = model.incidence_of(one, other).unwrap();
                assert!(verdict.meets(), "great circles {i} and {j} did not meet");
            }
        }
        // a circle with itself is not a meeting
        let err = message(model.incidence_of(&circles[0], &circles[0]));
        assert!(err.contains("coincident"), "{err}");
    }

    #[test]
    fn test_a_zero_normal_is_not_a_great_circle_and_antipodal_points_do_not_make_one() {
        // The two spherical non-degeneracies the planar kernel cannot have.
        // the zero direction is refused where it is built, since it is not a
        // point of the sphere at all
        let err = SRay::new(Q::ZERO, Q::ZERO, Q::ZERO).unwrap_err();
        assert!(err.to_string().contains("not a direction"), "{err}");
        // and a zero normal is refused by the great-circle constructor, which
        // is the empty circle rather than a line of the sphere
        let why = message(SphericalModel::great_circle(
            &SRay::new(Q::ZERO, Q::from_int(1), Q::ZERO).unwrap(),
        ));
        assert!(why.contains("not a line"), "{why}");
        // two antipodal rays lie on every great circle through their axis, so
        // they do not determine one
        let (one, antipode) = (
            SRay::new(Q::ONE, Q::ZERO, Q::ZERO).unwrap(),
            SRay::new(q(-1, 1), Q::ZERO, Q::ZERO).unwrap(),
        );
        assert!(one.antipodal(&antipode).unwrap());
        let why = message(SphericalModel.normal_of(&one, &antipode));
        assert!(why.contains("endpoints coincide"), "{why}");
    }

    #[test]
    fn test_spherical_incidence_through_a_lifted_planar_scene() {
        // The witness path the axiom report uses: a planar "parallel" pair,
        // lifted to great circles, which then meet -- so the parallel claim is
        // a contradiction in the spherical reading, with an exact witness.
        let graph = scene(
            "spherical",
            vec![kp("A", 0, 0), kp("B", 2, 0), kp("C", 0, 1), kp("D", 2, 1)],
            vec![Constraint::Parallel { first: seg("A", "B"), second: seg("C", "D") }],
        );
        let ray = SphericalModel::meet_of_segments(&graph, &seg("A", "B"), &seg("C", "D")).unwrap();
        // the direction exists and is exact; the report quotes it
        assert!(!ray.describe().is_empty());
        assert!(GeometryKind::Spherical.asserts(Axiom::AllLinesMeet));
        let report = GeometryKind::Spherical.check_axioms(&graph).unwrap();
        assert!(!report.is_consistent(), "a spherical parallel pair is a contradiction");
    }

    // -------------------------------------------- checking a scene's axioms --

    /// A scene of two horizontal segments one unit apart, marked `Parallel`.
    fn parallel_pair(geometry: &str) -> SceneGraph {
        scene(
            geometry,
            vec![kp("A", 0, 0), kp("B", 2, 0), kp("C", 0, 1), kp("D", 2, 1)],
            vec![Constraint::Parallel { first: seg("A", "B"), second: seg("C", "D") }],
        )
    }

    #[test]
    fn test_parallel_in_the_projective_plane_is_an_axiom_violation() {
        // The sharpest case in the whole module: a Euclidean figure with a
        // `Parallel` fact, read under a declared projective geometry. There,
        // any two lines meet, so the fact is a contradiction -- and the report
        // names the exact point where the two lines actually meet.
        let graph = parallel_pair("projective");
        let report = GeometryKind::Projective.check_axioms(&graph).unwrap();
        assert!(!report.is_consistent());
        // and it is *complete*: the statement was decided, and decided wrongly
        // for this environment. Completeness and consistency are different
        // questions, and a report that conflated them could not tell a caller
        // whether to delete the fact or go looking for more.
        assert!(report.is_complete(), "{report}");
        let violation = report
            .violation_for("parallel(AB, CD)")
            .expect("the parallel claim is a violation projectively");
        assert_eq!(violation.axiom, Axiom::AllLinesMeet);
        assert!(violation.detail.contains("meet at"), "{}", violation.detail);
        assert!(violation.detail.contains("[1 : 0 : 0]"), "the exact witness is missing: {}", violation.detail);
        // the report says which geometry it read under, and lists its axioms
        assert_eq!(report.geometry, GeometryKind::Projective);
        assert!(report.axioms.contains(&Axiom::AllLinesMeet));
    }

    #[test]
    fn test_parallel_in_the_affine_and_euclidean_planes_is_fine() {
        // The same figure, the same fact, two geometries where it holds. A
        // report that flagged this would be the over-application in its purest
        // form, so the negative case is tested as carefully as the positive.
        for name in ["euclidean", "affine"] {
            let kind: GeometryKind = name.parse().unwrap();
            let report = kind.check_axioms(&parallel_pair(name)).unwrap();
            assert!(report.is_consistent(), "{name}: {report}");
            assert!(report.is_complete(), "{name}: {report}");
            assert_eq!(report.violations.len(), 0, "{name}: {report}");
        }
    }

    #[test]
    fn test_parallel_in_the_hyperbolic_plane_is_underdetermined_not_false() {
        // The distinction the module exists for. In the hyperbolic plane a
        // "parallel" pair is not wrong -- many pairs do not meet -- but the
        // claim does not say which kind, so the report records it as
        // undecided. Reporting it as a violation would train a reasoner to
        // delete true statements of a non-Euclidean figure.
        let report = GeometryKind::Hyperbolic.check_axioms(&parallel_pair("hyperbolic")).unwrap();
        assert!(report.is_consistent(), "a hyperbolic parallel pair is not a contradiction");
        assert!(!report.is_complete(), "but it is not decided either: {report}");
        let open = report
            .underdetermination_for("parallel(AB, CD)")
            .expect("the parallel claim is underdetermined hyperbolically");
        assert_eq!(open.axiom, Axiom::InfinitelyManyParallelsThroughPoint);
        assert!(open.needed.contains("ultraparallel"), "{}", open.needed);
    }

    #[test]
    fn test_a_metric_claim_in_a_non_metric_geometry_is_refused() {
        // An equal-length fact is not a statement of the projective plane, and
        // accepting it would put a Euclidean assumption into a geometry that
        // has no metric to hold it.
        let graph = scene(
            "projective",
            vec![kp("A", 0, 0), kp("B", 3, 4), kp("C", 0, 0), kp("D", 1, 0)],
            vec![Constraint::EqualLength { first: seg("A", "B"), second: seg("C", "D") }],
        );
        let report = GeometryKind::Projective.check_axioms(&graph).unwrap();
        let violation = report
            .violation_for("len(AB)=len(CD)")
            .expect("a length claim is a violation projectively");
        assert_eq!(violation.axiom, Axiom::MetricDefined);
        assert!(violation.detail.contains("no metric"), "{}", violation.detail);
        // the same fact is fine where there is a metric
        let euclidean = scene(
            "euclidean",
            graph.points.clone(),
            vec![Constraint::EqualLength { first: seg("A", "B"), second: seg("C", "D") }],
        );
        assert!(GeometryKind::Euclidean.check_axioms(&euclidean).unwrap().is_consistent());
    }

    #[test]
    fn test_betweenness_in_a_projective_or_spherical_geometry_is_refused() {
        // A projective line is a circle and a great circle is closed, so one of
        // three collinear points is not between the other two: "between" names
        // a relation these geometries do not define.
        for name in ["projective", "spherical"] {
            let kind: GeometryKind = name.parse().unwrap();
            let graph = scene(
                name,
                vec![kp("A", 0, 0), kp("B", 2, 0), kp("M", 1, 0)],
                vec![Constraint::Between { a: "A".into(), m: "M".into(), b: "B".into() }],
            );
            let report = kind.check_axioms(&graph).unwrap();
            let violation = report
                .violation_for("M lies between A and B")
                .unwrap_or_else(|| panic!("{name} should refuse betweenness: {report}"));
            assert_eq!(violation.axiom, Axiom::BetweennessOrdered);
            // and it is fine in the three ordered geometries
            for ordered in ["euclidean", "affine", "hyperbolic"] {
                let graph = scene(
                    ordered,
                    vec![kp("A", 0, 0), kp("B", 2, 0), kp("M", 1, 0)],
                    vec![Constraint::Between { a: "A".into(), m: "M".into(), b: "B".into() }],
                );
                let ordered_kind: GeometryKind = ordered.parse().unwrap();
                assert!(
                    ordered_kind.check_axioms(&graph).unwrap().is_consistent(),
                    "{ordered} does have betweenness"
                );
            }
        }
    }

    #[test]
    fn test_checking_a_scene_under_the_wrong_geometry_is_refused() {
        // The compatibility check: a projective report on a scene that declares
        // itself spherical is not computed at all, because the two readings
        // would answer different questions.
        let graph = parallel_pair("spherical");
        let why = refusal(GeometryKind::Projective.check_axioms(&graph));
        match why {
            NonEuclideanError::GeometryMismatch { declared, scene } => {
                assert_eq!(declared, GeometryKind::Projective);
                assert_eq!(scene, "spherical");
            }
            other => panic!("expected a geometry mismatch, got {other:?}"),
        }
        // and an unknown declared geometry is an error, never a default
        let odd = scene("non-euclidean-ish", vec![kp("A", 0, 0)], vec![]);
        assert!(GeometryKind::of_scene(&odd).is_err());
    }

    #[test]
    fn test_a_report_lists_the_environment_and_its_findings() {
        // The report carries the whole environment, not only the part that was
        // upset: a reader of a violation needs to know which geometry produced
        // it, and which axioms that geometry had.
        let report = GeometryKind::Projective.check_axioms(&parallel_pair("projective")).unwrap();
        assert_eq!(report.axioms, GeometryKind::Projective.axioms());
        assert_eq!(report.checked.len(), 0, "the parallel fact was not accepted as checked");
        let text = report.to_string();
        assert!(text.contains("projective geometry"), "{text}");
        assert!(text.contains("violation"), "{text}");
        // and a report round-trips through serde, so it can be filed next to
        // the scene it came from
        let json = serde_json::to_string(&report).unwrap();
        let back: AxiomReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report);
    }

    #[test]
    fn test_an_observed_statement_is_not_a_violation() {
        // The kernel's confidence rule carries over: a fact below full
        // confidence is a diagram's guess, and this report does not get to call
        // a guess a contradiction.
        let mut graph = parallel_pair("projective");
        graph.facts[0] = Fact::observed(
            Constraint::Parallel { first: seg("A", "B"), second: seg("C", "D") },
            "diagram_grader",
            0.83,
        );
        let report = GeometryKind::Projective.check_axioms(&graph).unwrap();
        assert!(report.is_consistent(), "{report}");
        assert!(report.checked.is_empty(), "an observed statement was checked as a claim");
    }

    // ------------------------------------------- exactness vs evidence --

    #[test]
    fn test_an_angle_sum_cannot_become_a_fact() {
        // The honesty rule as an API distinction. Three exact cosines of 1/2 --
        // the angles of an equilateral triangle, and exact inputs -- sum to
        // 180 only in a Euclidean plane, and the sum is not a rational number
        // even there. So the measurement is evidence, and the attempt to make
        // it a fact is refused.
        let angles = vec![
            Angle3::new("A", "B", "C"),
            Angle3::new("B", "C", "A"),
            Angle3::new("C", "A", "B"),
        ];
        let cosines = vec![
            QSqrt::rational(Frac::from_q(q(1, 2))),
            QSqrt::rational(Frac::from_q(q(1, 2))),
            QSqrt::rational(Frac::from_q(q(1, 2))),
        ];
        let evidence = NumericEvidence::angle_sum_degrees(&angles, &cosines).unwrap();
        assert_eq!(evidence.unit, "degrees");
        assert!((evidence.value - 180.0).abs() < 1e-9, "{}", evidence.value);
        // the exact cosines went in and the exact cosines are still exact; it
        // is the sum that is not
        assert_eq!(cosines[0].rational_value().unwrap(), Some(q(1, 2)));
        // and the refusal is the enforcement
        let why = refusal(evidence.into_fact());
        match why {
            NonEuclideanError::EvidenceIsNotAFact { quantity } => {
                assert!(quantity.contains("angle sum"), "{quantity}");
            }
            other => panic!("expected an evidence refusal, got {other:?}"),
        }
    }

    #[test]
    fn test_evidence_reports_agreement_without_proving_it() {
        // The same measurement, read under each declared angle-sum axiom. This
        // is a consistency note and is offered as an `Option`, because two of
        // the five geometries declare no angle axiom at all and an honest
        // `false` there would read as a refutation.
        let evidence = NumericEvidence::degrees("angle sum of a triangle", 179.6);
        assert_eq!(evidence.agrees_with(GeometryKind::Euclidean), Some(true));
        assert_eq!(evidence.agrees_with(GeometryKind::Hyperbolic), Some(true));
        assert_eq!(evidence.agrees_with(GeometryKind::Spherical), Some(false));
        assert_eq!(evidence.agrees_with(GeometryKind::Projective), None);
        assert_eq!(evidence.agrees_with(GeometryKind::Affine), None);
        // a sum far enough from 180 disagrees with the Euclidean axiom, and the
        // half-degree tolerance is not wide enough to hide it
        let wrong = NumericEvidence::degrees("angle sum of a triangle", 150.0);
        assert_eq!(wrong.agrees_with(GeometryKind::Euclidean), Some(false));
        // and a spherical triangle's sum, more than 180, disagrees with the
        // Euclidean axiom in the way the geometry says it should
        let excess = NumericEvidence::degrees("angle sum of a spherical triangle", 200.0);
        assert_eq!(excess.agrees_with(GeometryKind::Spherical), Some(true));
        assert_eq!(excess.agrees_with(GeometryKind::Euclidean), Some(false));
        // the caveat travels with the number, so a report that quotes one
        // quotes the other
        assert!(excess.to_string().contains("not a fact"), "{excess}");
    }

    #[test]
    fn test_evidence_round_trips_through_serde_and_keeps_its_unit() {
        let evidence = NumericEvidence::of("arc length", 3.25, "radians");
        let json = serde_json::to_string(&evidence).unwrap();
        let back: NumericEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(back, evidence);
        assert_eq!(back.unit, "radians");
        // a non-degree quantity has no angle axiom to be consistent with, and
        // says so rather than comparing radians against degrees
        assert_eq!(evidence.agrees_with(GeometryKind::Euclidean), None);
    }

    #[test]
    fn test_hyperbolic_trigonometry_is_refused_rather_than_approximated() {
        // The hyperbolic sine and cosine of a rational length are in neither
        // the rationals nor the quadratic field, so there is no exact answer to
        // return here and the module says so instead of producing a float.
        for operation in ["sine", "cosine", "distance"] {
            let why = refusal(hyperbolic_trig(operation));
            match why {
                NonEuclideanError::OutOfScope { detail } => {
                    assert!(detail.contains("hyperbolic"), "{detail}");
                    assert!(detail.contains("neither the rationals nor"), "{detail}");
                    assert!(detail.contains(operation), "the refusal does not name {operation}: {detail}");
                }
                other => panic!("expected an out-of-scope refusal for {operation}, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_incidence_is_a_decision_and_its_arms_are_all_exact() {
        // The three arms of the shared verdict type, each reached in the model
        // that produces it, and each naming an exact point or the absence of
        // one. No arm is "probably".
        let graph = scene(
            "affine",
            vec![kp("A", 0, 0), kp("B", 2, 0), kp("C", 0, 1), kp("D", 2, 1)],
            vec![],
        );
        // crossing lines meet inside, at the exact point (1, 1/2)
        let crossing = ProjectivePlane::incidence_of(&graph, &seg("A", "D"), &seg("B", "C")).unwrap();
        assert!(crossing.meets(), "{crossing}");
        assert_eq!(crossing.at().unwrap().to_affine().unwrap(), (Q::ONE, Q::new(1, 2).unwrap()));
        // parallel ones meet at an ideal point, which is a meeting too
        let parallel = ProjectivePlane::incidence_of(&graph, &seg("A", "B"), &seg("C", "D")).unwrap();
        assert!(matches!(parallel, Incidence::MeetAtInfinity(_)), "{parallel}");
        assert!(!parallel.meets(), "a parallel pair does not meet inside the plane");
        assert!(parallel.at().is_some(), "but the ideal point is named");
        // and a line with itself is not a meeting at all
        let coincident = message(ProjectivePlane::incidence_of(&graph, &seg("A", "B"), &seg("A", "B")));
        assert!(coincident.contains("coincident"), "{coincident}");
        // and a hyperbolic ultraparallel pair has no point at all
        let disk = klein();
        let axis = PLine::new(Q::ZERO, Q::ONE, Q::ZERO).unwrap();
        let off = PPoint::affine(Q::ZERO, Q::new(1, 2).unwrap());
        let witness = disk.non_merging_lines_through(&off, &axis, 1).unwrap().remove(0);
        let disjoint = disk.incidence_of(&axis, &witness).unwrap();
        assert!(!disjoint.meets());
        assert!(disjoint.at().is_none());
        assert!(disjoint.to_string().contains("no point"), "{disjoint}");
    }

    #[test]
    fn test_every_public_geometry_name_round_trips_through_the_scene_field() {
        // The end-to-end path a file takes: a scene declares a geometry by
        // name, the engine parses it, and the axioms and the permitted
        // theorems follow from that string alone.
        for kind in GeometryKind::all() {
            let graph = scene(kind.as_str(), vec![kp("A", 0, 0)], vec![]);
            let parsed = GeometryKind::of_scene(&graph).unwrap();
            assert_eq!(parsed, *kind);
            assert!(parsed.matches_scene(&graph));
            assert!(parsed.check_scene(&graph).is_ok());
            assert_eq!(parsed.axioms(), kind.axioms());
            assert_eq!(parsed.valid_theorems(), kind.valid_theorems());
            // a scene that says nothing at all is Euclidean, the kernel's own
            // default, and this module reads it as the Euclidean environment
            let default = graph_from_points(vec![kp("A", 0, 0)], vec![]);
            assert_eq!(default.geometry, "euclidean");
            assert_eq!(GeometryKind::of_scene(&default).unwrap(), GeometryKind::Euclidean);
        }
    }

    #[test]
    fn test_scene_constraints_route_to_the_axioms_that_decide_them() {
        // Each constraint family lands on the axiom the environment either has
        // or lacks. The interesting assertions are the *negatives*: the
        // geometry that does have the axiom is not asked to have it, and the
        // one that lacks it is not left with a `bool` the caller has to
        // interpret.
        let right = || Constraint::RightAngle { at: Angle3::new("A", "B", "C") };
        let cases: Vec<(Constraint, &str, Option<Axiom>)> = vec![
            (right(), "euclidean", Some(Axiom::AngleMeasureExists)),
            (right(), "projective", None),
            (
                Constraint::EqualLength { first: seg("A", "B"), second: seg("C", "D") },
                "affine",
                None,
            ),
            (
                Constraint::Between { a: "A".into(), m: "M".into(), b: "B".into() },
                "spherical",
                None,
            ),
        ];
        for (constraint, geometry, needed) in cases {
            let graph = scene(geometry, vec![kp("A", 0, 0), kp("B", 1, 0), kp("C", 0, 1), kp("D", 1, 1), kp("M", 1, 2)], vec![]);
            let mut graph = graph;
            graph.facts.push(Fact::given(constraint.clone()));
            let kind: GeometryKind = geometry.parse().unwrap();
            let report = kind.check_axioms(&graph).unwrap();
            match needed {
                Some(axiom) => {
                    assert!(kind.asserts(axiom), "{kind} should hold {axiom}");
                    assert!(report.is_consistent(), "{kind} should accept {constraint:?}: {report}");
                    assert_eq!(report.checked.len(), 1);
                }
                None => {
                    assert!(!report.is_consistent(), "{kind} should refuse {constraint:?}: {report}");
                    assert_eq!(report.violations.len(), 1, "{report}");
                }
            }
        }
    }
}
