//! Structure-preserving augmentation (roadmap Phase 33, audit fix 10).
//!
//! Pattern recognition and structural understanding look identical on a test
//! set of one figure, so the training distribution has to make them differ. The
//! cheap version of that -- relabel the vertices, keep the labels' meaning -- is
//! not augmentation at all; it is duplication. This module instead applies the
//! transformations that preserve the *proof graph* while destroying the surface
//! form: rotation, reflection, translation, scaling, vertex permutation, vertex
//! renaming, and a shear (an affine change of coordinates that keeps incidence,
//! parallelism, ratios along a line and area ratios, and destroys lengths and
//! angles).
//!
//! Two things make the family honest:
//!
//! - the exact transformations (rotation by a rational cosine/sine pair,
//!   reflection, translation, uniform scaling) must leave every derived fact
//!   true, which [`invariance_report`] verifies predicate by predicate over the
//!   kernel's rationals and reports as a confusion matrix rather than trusting;
//! - the affine shear must *refute* length- and angle-bearing facts. A model
//!   that answers a sheared figure with the unstretched answer was reading a
//!   memorized template, and the report says so.
//!
//! Facts are carried over by transformation of the coordinates plus a
//! predicate-preserving remap of the names; the underlying proof graph is the
//! object being augmented, and the coordinates are its current presentation.

//! # What lives here
//!
//! - [`PredicateClass`] and [`classify`]: which family a predicate belongs to
//!   -- incidence, parallelism, ratio, length, angle, area, congruence -- so
//!   that "the shear preserved the structure" is a countable claim rather than
//!   an adjective.
//! - [`Transform`]: the six presentations of one proof graph. The first five
//!   are similarities and are claimed to preserve *every* derived fact; the
//!   sixth, [`Transform::Shear`], is claimed to preserve incidence, parallelism,
//!   ratios along a line and areas while refuting lengths and angles.
//! - [`Transform::apply`]: the transform, and [`Transform::apply_reported`]: the
//!   transform plus a [`TransformReport`] naming every fact it kept, every fact
//!   it dropped and *why*, the scene's confidence before and after, and the
//!   per-predicate-class tally. Facts are re-decided by the kernel's own
//!   [`constraint_holds_in`], never copied on the transform's say-so.
//! - [`invariance_report`]: the family run against a scene, predicate class by
//!   predicate class, producing a confusion matrix ([`ClassCell`]) and a
//!   [`InvarianceReport::verdict`] that says plainly whether each family
//!   behaved as claimed.
//! - [`augmentations`]: `n` *distinct* surface presentations of one proof graph
//!   -- the same structure as `ABC`, as `BCA`, reflected, rotated, scaled --
//!   each paired with the answer it should elicit ([`AugmentedExample`]).
//!
//! # The non-degeneracy discipline
//!
//! Every transform is refused rather than half-applied. A rotation whose pair
//! is not on the unit circle is not a rotation but a shear wearing a rotation's
//! name, and it is refused with [`GeometryError::NoRationalSolution`]; a zero
//! or negative scale factor is refused with [`GeometryError::EmptyGeometry`]; a
//! reflection axis through two coincident points is
//! [`GeometryError::DegenerateSegment`]; a vertex permutation that is not a
//! bijection of the scene's points is refused by name. Nothing here reaches for
//! a float to make a transform "nearly" work.
//!
//! # What this module does not claim
//!
//! The rotation catalogue is the rational-cosine one: quarter turns, half
//! turns and 3-4-5 turns are exact, and a 60-degree turn is *absent* rather
//! than approximated, because its sine is `sqrt(3)/2` and a rounded coordinate
//! would be a lie the rest of the pipeline would then treat as a premise. The
//! shear implemented here is the horizontal one `x' = x + k y`, of determinant
//! exactly `1`, so areas are preserved; a general affine map is not provided,
//! and a catalogue of exact transforms is finite, so an invariance report
//! repeats its catalogue when asked for more trials than the catalogue holds
//! rather than inventing transforms it cannot decide.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

use crate::geomkernel::{
    constraint_holds_in, Angle3, Constraint, Fact, Frac, GeometryError, KCircle, KPoint,
    Provenance, SceneGraph, Segment, Tri3, Q,
};

// --------------------------------------------------------- predicate class --

/// The family a predicate belongs to, and therefore the family of claims an
/// augmentation has to respect.
///
/// The classes are not a partition chosen for elegance; each one is the set of
/// predicates a named transform family is *claimed* to leave alone, and that
/// claim is what [`invariance_report`] counts. Putting `Perpendicular` in
/// [`PredicateClass::Angle`] rather than somewhere of its own is the honest
/// call: a perpendicularity is a right angle, and a shear does not preserve
/// right angles. Putting `MidpointOf` and `RatioOf` in
/// [`PredicateClass::Ratio`] rather than in `Incidence` is the same kind of
/// call: a midpoint is not where a point *is*, it is how a point *divides* a
/// segment, and an affine map preserves exactly that.
///
/// Circles are classed with lengths, not with incidences, and the reason is
/// worth stating because it is the one place the classification looks
/// surprising. `OnCircle` and `Diameter` are statements about a radius: a shear
/// maps a circle to an ellipse, the kernel has no ellipse, and the honest
/// outcome is that the object and every incidence on it cannot be carried --
/// reported as a metric fact lost, not as an incidence broken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredicateClass {
    /// Where objects are: collinearity, betweenness, distinctness, a genuine
    /// triangle. Preserved by every transform here.
    Incidence,
    /// Two directions agreeing. Preserved by every transform here, affine or
    /// not.
    Parallelism,
    /// How a point divides a segment -- `MidpointOf`, `RatioOf`. Affine
    /// invariant, and the reason a shear is a useful augmentation at all.
    Ratio,
    /// A metric claim: a squared length, a comparison of lengths, a radius.
    Length,
    /// A claim about an angle's cosine: right angles, angle equality, a named
    /// exact cosine.
    Angle,
    /// Equality of two areas. Preserved by the determinant-one shear here, and
    /// by every similarity.
    Area,
    /// Two triangles with three equal sides. The most fragile claim in the
    /// library: a similarity keeps it, a shear does not.
    Congruence,
}

impl PredicateClass {
    /// Every class, in declaration order -- the order the report prints them
    /// in, so a matrix read twice reads the same way.
    pub const ALL: [PredicateClass; 7] = [
        PredicateClass::Incidence,
        PredicateClass::Parallelism,
        PredicateClass::Ratio,
        PredicateClass::Length,
        PredicateClass::Angle,
        PredicateClass::Area,
        PredicateClass::Congruence,
    ];

    /// A one-word name for the class, for a table row or a drop reason.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Incidence => "incidence",
            Self::Parallelism => "parallelism",
            Self::Ratio => "ratio",
            Self::Length => "length",
            Self::Angle => "angle",
            Self::Area => "area",
            Self::Congruence => "congruence",
        }
    }

    /// The class's position in [`PredicateClass::ALL`], which is what a tally
    /// vector is indexed by. Every class has a position: the vector is built
    /// from `ALL`, so a class without an index would be a class whose tally
    /// could not be recorded.
    pub fn index(self) -> usize {
        match self {
            Self::Incidence => 0,
            Self::Parallelism => 1,
            Self::Ratio => 2,
            Self::Length => 3,
            Self::Angle => 4,
            Self::Area => 5,
            Self::Congruence => 6,
        }
    }
}

impl fmt::Display for PredicateClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.describe())
    }
}

/// The class of a predicate: the one place the classification is defined, so
/// the augmenter's tally and the report's verdict cannot disagree about what a
/// claim is made of.
///
/// Every one of the kernel's [`Constraint`] variants is covered here, and a new
/// variant added to the kernel will not compile until someone says which family
/// it belongs to -- which is the point of a match rather than a lookup table.
pub fn classify(constraint: &Constraint) -> PredicateClass {
    match constraint {
        Constraint::Collinear { .. }
        | Constraint::Distinct { .. }
        | Constraint::NonCollinear { .. }
        | Constraint::Triangle { .. }
        | Constraint::Between { .. } => PredicateClass::Incidence,
        Constraint::Parallel { .. } => PredicateClass::Parallelism,
        Constraint::MidpointOf { .. } | Constraint::RatioOf { .. } => PredicateClass::Ratio,
        Constraint::Perpendicular { .. }
        | Constraint::AngleEqual { .. }
        | Constraint::RightAngle { .. }
        | Constraint::AngleIs { .. } => PredicateClass::Angle,
        Constraint::EqualLength { .. }
        | Constraint::LengthIs { .. }
        | Constraint::ScaleLength { .. }
        | Constraint::Circle { .. }
        | Constraint::OnCircle { .. }
        | Constraint::Diameter { .. } => PredicateClass::Length,
        Constraint::AreaEqual { .. } => PredicateClass::Area,
        Constraint::Congruent { .. } => PredicateClass::Congruence,
    }
}

impl Constraint {
    /// The class of this predicate. A method on the kernel's own type so a
    /// caller holding a [`Constraint`] need not remember to go and look.
    pub fn classify(&self) -> PredicateClass {
        classify(self)
    }
}

// -------------------------------------------------------- transform family --

/// Which family of the augmentation library a transform belongs to: the five
/// families claimed to preserve every fact, and the one claimed to break the
/// metric ones.
///
/// A row of the confusion matrix is a family, not a single transform, because
/// the claim is about the family. A quarter turn and a 3-4-5 turn are both
/// "rotation"; the report says whether rotation preserved incidence, and it
/// says it for every turn it ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransformFamily {
    /// A rational rotation. A similarity.
    Rotation,
    /// A reflection in a line. A similarity, though orientation-reversing.
    Reflection,
    /// A translation. The similarity that leaves every distance alone.
    Translation,
    /// A uniform rational scaling. A similarity.
    Scaling,
    /// A relabelling of the vertices. A symmetry of the *proof graph*, and the
    /// clearest demonstration that the graph, not the drawing, is the object.
    Permutation,
    /// The horizontal shear `x' = x + k y`. The one affine, non-similarity
    /// here: incidence, parallelism, ratios and areas survive; lengths and
    /// angles do not.
    Shear,
}

impl TransformFamily {
    /// Every family, in report order.
    pub const ALL: [TransformFamily; 6] = [
        TransformFamily::Rotation,
        TransformFamily::Reflection,
        TransformFamily::Translation,
        TransformFamily::Scaling,
        TransformFamily::Permutation,
        TransformFamily::Shear,
    ];

    /// A one-word name, for a table row.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Rotation => "rotation",
            Self::Reflection => "reflection",
            Self::Translation => "translation",
            Self::Scaling => "scaling",
            Self::Permutation => "permutation",
            Self::Shear => "shear",
        }
    }

    /// Whether the family is claimed to preserve *every* derived fact. The
    /// shear is not, and every other family here is: each of the five is either
    /// a similarity, which leaves lengths scaled by one factor and angles and
    /// incidences alone, or a relabelling, which changes nothing at all.
    pub fn is_exact(self) -> bool {
        !matches!(self, TransformFamily::Shear)
    }
}

impl fmt::Display for TransformFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.describe())
    }
}

// --------------------------------------------------------------- transform --

/// A reflection axis: either a line through two named points of the scene, or
/// one of the coordinate axes.
///
/// A line is a pair of distinct points in this kernel, so the through-points
/// form is not a special case bolted on afterwards -- it is the same definition
/// the rest of the kernel uses, which means the axis is named the same way a
/// segment is and cannot be half-specified.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "axis", rename_all = "snake_case")]
pub enum ReflectionAxis {
    /// The line through two named points of the scene.
    ThroughPoints { a: String, b: String },
    /// The line `y = 0`.
    XAxis,
    /// The line `x = 0`.
    YAxis,
}

impl ReflectionAxis {
    /// A one-line description, for a report or a drop reason.
    pub fn describe(&self) -> String {
        match self {
            Self::ThroughPoints { a, b } => format!("the line {a}{b}"),
            Self::XAxis => "the x axis".to_string(),
            Self::YAxis => "the y axis".to_string(),
        }
    }
}

/// A change of presentation of one proof graph.
///
/// The first five variants are similarities (or, for [`Transform::Permutation`],
/// a symmetry of the graph itself) and carry the claim *every* derived fact
/// survives. [`Transform::Shear`] carries the opposite claim, and the module's
/// whole discipline is that the two claims are checked rather than asserted:
/// [`invariance_report`] re-decides every predicate after every trial and
/// publishes the tally.
///
/// Nothing here holds an angle. A rotation is a rational `(cos, sin)` pair,
/// checked to satisfy `cos^2 + sin^2 == 1` exactly, because a rotation whose
/// pair is off the unit circle is not a rotation -- it is a shear with a
/// misleading name, and it is refused rather than applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transform {
    /// A rotation by an exact rational cosine and sine.
    Rotation { cos: Q, sin: Q },
    /// A reflection in a line.
    Reflection { axis: ReflectionAxis },
    /// A translation by an exact rational vector.
    Translation { dx: Q, dy: Q },
    /// A uniform scaling by the positive rational `num / den`. The two parts
    /// are stored separately because a uniform scale is a *positive* similarity
    /// and the positivity has to be visible in the value, not implied.
    Scaling { num: Q, den: Q },
    /// A relabelling of the vertices: an old name mapped to a new one.
    /// Renaming the drawing without touching the mathematics -- which is
    /// precisely the augmentation the module doc calls duplication, kept here
    /// only as the identity the other transforms are measured against.
    Permutation { rename: BTreeMap<String, String> },
    /// The horizontal shear `(x, y) -> (x + k y, y)`. Invertible for every
    /// `k`, and not a similarity for any `k != 0`.
    Shear { k: Q },
}

impl Transform {
    /// The quarter turn: `cos = 0`, `sin = 1`. Exact, and the one rotation
    /// that maps the lattice to itself.
    pub fn quarter_turn() -> Self {
        Self::Rotation {
            cos: Q::ZERO,
            sin: Q::ONE,
        }
    }

    /// The half turn: `cos = -1`, `sin = 0`.
    pub fn half_turn() -> Self {
        Self::Rotation {
            cos: Q::from_int(-1),
            sin: Q::ZERO,
        }
    }

    /// The 3-4-5 turn: `cos = 3/5`, `sin = 4/5`. The rotation that is not a
    /// familiar angle -- no degree value anyone memorizes -- and still exact.
    pub fn three_four_five_turn() -> Self {
        Self::Rotation {
            cos: Q { num: 3, den: 5 },
            sin: Q { num: 4, den: 5 },
        }
    }

    /// A rotation by the given exact cosine and sine. Refuses a pair that does
    /// not satisfy `cos^2 + sin^2 == 1` exactly: the off-unit-circle pair is
    /// the one that would silently shear the figure while claiming to turn it.
    pub fn rotation(cos: Q, sin: Q) -> anyhow::Result<Self> {
        let total = cos.mul(&cos)?.add(&sin.mul(&sin)?)?;
        anyhow::ensure!(
            total == Q::ONE,
            GeometryError::NoRationalSolution(format!(
                "cos {cos} and sin {sin} do not satisfy cos^2 + sin^2 = 1 (got {total}), so they \
                 are not a rotation"
            ))
        );
        Ok(Self::Rotation { cos, sin })
    }

    /// A translation by `(dx, dy)`. Any rational vector, including zero; the
    /// identity is a legitimate (if uninformative) presentation.
    pub fn translation(dx: Q, dy: Q) -> Self {
        Self::Translation { dx, dy }
    }

    /// A uniform scaling by `num / den`. Refuses a zero scale -- it collapses
    /// every point onto the origin and there is no figure left to reason about
    /// -- and refuses a negative one, which is a half turn wearing a scale's
    /// name and is available as [`Transform::half_turn`].
    pub fn scaling(num: Q, den: Q) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !den.is_zero(),
            GeometryError::NoRationalSolution(
                "a scaling with a zero denominator is not a number".to_string()
            )
        );
        let factor = num.div(&den)?;
        anyhow::ensure!(
            !factor.is_zero(),
            GeometryError::EmptyGeometry(
                "a zero scale factor collapses the whole figure onto one point".to_string()
            )
        );
        anyhow::ensure!(
            Q::ZERO.less(&factor),
            GeometryError::NoRationalSolution(format!(
                "a scale factor of {factor} is negative; that is a half turn, not a scaling, and \
                 Transform::half_turn is where it belongs"
            ))
        );
        Ok(Self::Scaling { num, den })
    }

    /// The shear `(x, y) -> (x + k y, y)`. A zero `k` is the identity and is
    /// allowed, because it is not degenerate -- it just teaches nothing.
    pub fn shear(k: Q) -> Self {
        Self::Shear { k }
    }

    /// A relabelling. The map must be a bijection on the scene's points; a map
    /// that is not gets refused when it is applied, not when it is built,
    /// because whether a rename is legal depends on the scene it is applied to.
    pub fn permutation(rename: BTreeMap<String, String>) -> Self {
        Self::Permutation { rename }
    }

    /// Reflection in the x axis: `(x, y) -> (x, -y)`.
    pub fn reflection_over_x_axis() -> Self {
        Self::Reflection {
            axis: ReflectionAxis::XAxis,
        }
    }

    /// Reflection in the y axis: `(x, y) -> (-x, y)`.
    pub fn reflection_over_y_axis() -> Self {
        Self::Reflection {
            axis: ReflectionAxis::YAxis,
        }
    }

    /// Reflection in the line through two named points.
    pub fn reflection_over_points(a: &str, b: &str) -> Self {
        Self::Reflection {
            axis: ReflectionAxis::ThroughPoints {
                a: a.to_string(),
                b: b.to_string(),
            },
        }
    }

    /// The family this transform belongs to.
    pub fn family(&self) -> TransformFamily {
        match self {
            Self::Rotation { .. } => TransformFamily::Rotation,
            Self::Reflection { .. } => TransformFamily::Reflection,
            Self::Translation { .. } => TransformFamily::Translation,
            Self::Scaling { .. } => TransformFamily::Scaling,
            Self::Permutation { .. } => TransformFamily::Permutation,
            Self::Shear { .. } => TransformFamily::Shear,
        }
    }

    /// A one-line description, for a report, a ledger note, or an error.
    pub fn describe(&self) -> String {
        match self {
            Self::Rotation { cos, sin } => format!("rotation by (cos {cos}, sin {sin})"),
            Self::Reflection { axis } => format!("reflection in {}", axis.describe()),
            Self::Translation { dx, dy } => format!("translation by ({dx}, {dy})"),
            Self::Scaling { num, den } => format!("scaling by {num}/{den}"),
            Self::Permutation { rename } => format!(
                "permutation of {} vertices ({})",
                rename.len(),
                rename
                    .iter()
                    .map(|(from, to)| format!("{from}->{to}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Shear { k } => format!("shear with k = {k}"),
        }
    }

    /// Whether the transform leaves every derived fact true, as a claim about
    /// the family rather than about one instance. The shear does not, and the
    /// report treats its length and angle facts as *expected losses*.
    pub fn preserves_every_fact(&self) -> bool {
        self.family().is_exact()
    }

    /// Apply the transform, returning the transformed scene.
    pub fn apply(&self, graph: &SceneGraph) -> anyhow::Result<SceneGraph> {
        Ok(self.apply_reported(graph)?.graph)
    }

    /// Apply the transform and report what happened to every fact.
    pub fn apply_reported(&self, graph: &SceneGraph) -> anyhow::Result<TransformReport> {
        self.checked(graph)
    }
}

impl fmt::Display for Transform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.describe())
    }
}

impl Serialize for Transform {
    /// Serialized through [`Frac`], because `Q` is an arithmetic type and the
    /// kernel's JSON form of a rational is `Frac`. The round trip is exact:
    /// `Frac::to_q` renormalizes, so a deserialized transform is a rational
    /// the kernel can reason about rather than a pair of integers hoping.
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        TransformWire::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Transform {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // A transform read off disk is validated exactly like one built in
        // code: a JSON file carrying `cos^2 + sin^2 != 1` is refused here, at
        // the boundary, rather than being applied to a figure and quietly
        // shearing it. The boundary is the only place this can be enforced
        // once.
        let wire = TransformWire::deserialize(deserializer)?;
        Transform::try_from(wire).map_err(serde::de::Error::custom)
    }
}

/// The JSON shape of a [`Transform`]: the same six families, with each rational
/// as a [`Frac`], which is the only rational form the kernel serializes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TransformWire {
    Rotation { cos: Frac, sin: Frac },
    Reflection { axis: ReflectionAxis },
    Translation { dx: Frac, dy: Frac },
    Scaling { num: Frac, den: Frac },
    Permutation { rename: BTreeMap<String, String> },
    Shear { k: Frac },
}

impl From<&Transform> for TransformWire {
    fn from(transform: &Transform) -> Self {
        match transform {
            Transform::Rotation { cos, sin } => TransformWire::Rotation {
                cos: Frac::from_q(*cos),
                sin: Frac::from_q(*sin),
            },
            Transform::Reflection { axis } => TransformWire::Reflection { axis: axis.clone() },
            Transform::Translation { dx, dy } => TransformWire::Translation {
                dx: Frac::from_q(*dx),
                dy: Frac::from_q(*dy),
            },
            Transform::Scaling { num, den } => TransformWire::Scaling {
                num: Frac::from_q(*num),
                den: Frac::from_q(*den),
            },
            Transform::Permutation { rename } => TransformWire::Permutation {
                rename: rename.clone(),
            },
            Transform::Shear { k } => TransformWire::Shear {
                k: Frac::from_q(*k),
            },
        }
    }
}

impl TryFrom<TransformWire> for Transform {
    type Error = anyhow::Error;

    fn try_from(wire: TransformWire) -> Result<Self, Self::Error> {
        match wire {
            TransformWire::Rotation { cos, sin } => Self::rotation(cos.to_q()?, sin.to_q()?),
            TransformWire::Reflection { axis } => Ok(Self::Reflection { axis }),
            TransformWire::Translation { dx, dy } => Ok(Self::Translation {
                dx: dx.to_q()?,
                dy: dy.to_q()?,
            }),
            TransformWire::Scaling { num, den } => Self::scaling(num.to_q()?, den.to_q()?),
            TransformWire::Permutation { rename } => Ok(Self::Permutation { rename }),
            TransformWire::Shear { k } => Ok(Self::Shear { k: k.to_q()? }),
        }
    }
}

/// The affine part of a transform, as an exact 2x2 matrix and an exact offset:
/// `p' = [[a, b], [c, d]] p + (tx, ty)`.
///
/// One representation for all six variants, for one reason. Whether a transform
/// preserves lengths is a property of this matrix's determinant and of whether
/// it is a similarity -- and having the matrix in hand is what lets the circle
/// bookkeeping and the degenerate checks be written once, exactly, rather than
/// six times, approximately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Affine {
    a: Q,
    b: Q,
    c: Q,
    d: Q,
    tx: Q,
    ty: Q,
}

impl Affine {
    /// The identity, which is what a permutation contributes to the geometry:
    /// it moves names and not locations, so its coordinate matrix is the
    /// identity and only the rename below does anything.
    fn identity() -> Self {
        Self {
            a: Q::ONE,
            b: Q::ZERO,
            c: Q::ZERO,
            d: Q::ONE,
            tx: Q::ZERO,
            ty: Q::ZERO,
        }
    }

    /// The exact determinant, `ad - bc`. Zero means the map collapses the
    /// plane onto a line, which is not a change of presentation of anything.
    fn determinant(&self) -> anyhow::Result<Q> {
        self.a.mul(&self.d)?.sub(&self.b.mul(&self.c)?)
    }

    /// The square of the length scale factor: `a^2 + b^2`, which for a
    /// similarity is `s^2` where `s` is how much it stretches a segment.
    ///
    /// This is the number a *squared* quantity has to be multiplied by, and
    /// getting it from the matrix rather than from the determinant is not
    /// fussiness. The determinant of a scaling by `f` is `f^2`, so squaring it
    /// again would give `f^4` and a circle's radius would grow twice too
    /// fast; `a^2 + b^2` is `f^2`, and it is also exactly `1` for a rotation
    /// (`cos^2 + sin^2`), for a reflection (`1 + 0`) and for a translation
    /// (`1 + 0`), which is precisely the claim those three families make about
    /// every length in the figure.
    fn squared_length_scale(&self) -> anyhow::Result<Q> {
        self.a.mul(&self.a)?.add(&self.b.mul(&self.b)?)
    }

    /// Whether the linear part is a similarity -- a scalar multiple of an
    /// orthogonal matrix, decided exactly by `a^2 + b^2 == c^2 + d^2 != 0`.
    /// This is the predicate the shear fails, and it is why the shear is the
    /// one variant in the library that drops metric facts.
    fn is_similarity(&self) -> anyhow::Result<bool> {
        let left = self.a.mul(&self.a)?.add(&self.b.mul(&self.b)?)?;
        let right = self.c.mul(&self.c)?.add(&self.d.mul(&self.d)?)?;
        Ok(left == right && !left.is_zero())
    }

    /// The image of an exact point under the affine map.
    fn map(&self, x: Q, y: Q) -> anyhow::Result<(Q, Q)> {
        let nx = self.a.mul(&x)?.add(&self.b.mul(&y)?)?.add(&self.tx)?;
        let ny = self.c.mul(&x)?.add(&self.d.mul(&y)?)?.add(&self.ty)?;
        Ok((nx, ny))
    }
}

impl Transform {
    /// The transform's exact affine part. `graph` is needed only for a
    /// reflection axis named by two points of the scene, because the line
    /// through two points is an object, not a constant.
    fn affine(&self, graph: &SceneGraph) -> anyhow::Result<Affine> {
        match self {
            // `(x, y) -> (x cos - y sin, x sin + y cos)`, so `a = cos`,
            // `b = -sin`, `c = sin`, `d = cos` and the determinant is
            // `cos^2 + sin^2 = 1` exactly -- which is what the constructor's
            // unit-circle check buys.
            Self::Rotation { cos, sin } => Ok(Affine {
                a: *cos,
                b: sin.neg()?,
                c: *sin,
                d: *cos,
                tx: Q::ZERO,
                ty: Q::ZERO,
            }),
            Self::Translation { dx, dy } => Ok(Affine {
                tx: *dx,
                ty: *dy,
                ..Affine::identity()
            }),
            Self::Scaling { num, den } => {
                let factor = num.div(den)?;
                Ok(Affine {
                    a: factor,
                    d: factor,
                    ..Affine::identity()
                })
            }
            Self::Shear { k } => Ok(Affine {
                b: *k,
                ..Affine::identity()
            }),
            Self::Permutation { .. } => Ok(Affine::identity()),
            Self::Reflection { axis } => match axis {
                ReflectionAxis::XAxis => Ok(Affine {
                    d: Q::from_int(-1),
                    ..Affine::identity()
                }),
                ReflectionAxis::YAxis => Ok(Affine {
                    a: Q::from_int(-1),
                    ..Affine::identity()
                }),
                ReflectionAxis::ThroughPoints { a, b } => {
                    let (ax, ay) = graph.coords(a)?;
                    let (bx, by) = graph.coords(b)?;
                    let (ux, uy) = (bx.sub(&ax)?, by.sub(&ay)?);
                    let norm = ux.mul(&ux)?.add(&uy.mul(&uy)?)?;
                    // The kernel's own refusal, for the kernel's own reason: a
                    // line needs a direction, and two coincident points do not
                    // have one.
                    anyhow::ensure!(!norm.is_zero(), GeometryError::DegenerateSegment);
                    let xx = ux.mul(&ux)?.div(&norm)?;
                    let xy = ux.mul(&uy)?.div(&norm)?;
                    let yy = uy.mul(&uy)?.div(&norm)?;
                    let two = Q::from_int(2);
                    // `p' = 2 proj_L(p) - p` with `proj_L(p) = a + ((p-a).u/u.u) u`.
                    // Written through `a` so the offset is exact and the
                    // reflection fixes `a` itself, as a reflection must.
                    let ma = xx.mul(&ax)?.add(&xy.mul(&ay)?)?;
                    let mb = xy.mul(&ax)?.add(&yy.mul(&ay)?)?;
                    Ok(Affine {
                        a: xx.mul(&two)?.sub(&Q::ONE)?,
                        b: xy.mul(&two)?,
                        c: xy.mul(&two)?,
                        d: yy.mul(&two)?.sub(&Q::ONE)?,
                        tx: ax.add(&ax)?.sub(&ma)?,
                        ty: ay.add(&ay)?.sub(&mb)?,
                    })
                }
            },
        }
    }

    /// The rename this transform performs, checked against the scene.
    ///
    /// Empty for the five geometric variants -- they move locations and keep
    /// names. For a permutation it is validated here rather than trusted: every
    /// source must be a declared point, two sources may not land on one name,
    /// and a target may not be a name that some *other* point is keeping,
    /// because a graph with two points called `A` is the exact configuration
    /// audit fix 11 exists to refuse.
    fn rename_map(&self, graph: &SceneGraph) -> anyhow::Result<BTreeMap<String, String>> {
        let Self::Permutation { rename } = self else {
            return Ok(BTreeMap::new());
        };
        let declared: Vec<String> = graph.points.iter().map(|p| p.name.clone()).collect();
        for from in rename.keys() {
            anyhow::ensure!(
                declared.contains(from),
                GeometryError::EmptyGeometry(format!(
                    "the permutation renames '{from}', which the scene does not declare"
                ))
            );
        }
        // Two sources reaching one target is the case that matters, and it has
        // to be counted over the *whole* map rather than checked pairwise: a
        // three-cycle `A->B, B->C, C->A` is legitimate and pairwise checking
        // would reject it, while `A->B, B->B` is not and pairwise checking
        // would let it through. So: how many sources claim each target, and a
        // target claimed twice is a refusal.
        for (from, to) in rename {
            let claimants = rename.iter().filter(|(_, other)| *other == to).count();
            anyhow::ensure!(
                claimants == 1,
                GeometryError::RepeatedPoint {
                    name: format!("'{to}', which '{from}' and another point both rename to"),
                }
            );
        }
        // A target that is a declared point the map *keeps* is also a collision:
        // the kept point is still there under that name. This is the case that
        // is not a two-sources-one-target at all -- `A->B` with `B` untouched
        // -- and it is exactly the configuration audit fix 11 exists to refuse.
        for (from, to) in rename {
            if from != to && declared.contains(to) && !rename.contains_key(to) {
                return Err(GeometryError::RepeatedPoint {
                    name: format!("'{to}', which '{from}' renames onto while '{to}' keeps it"),
                }
                .into());
            }
        }
        Ok(rename.clone())
    }

    /// Apply the transform, keeping only what survives an exact re-decision,
    /// and report what happened.
    ///
    /// This is the heart of the module and the place the "not relabeled
    /// duplicates" claim is either earned or not. Facts are never copied on the
    /// transform's say-so: every predicate is remapped (names for a
    /// permutation, the squared payload for a scaling) and then handed to the
    /// kernel's own [`constraint_holds_in`] against the *transformed*
    /// coordinates. A predicate that comes back false is dropped, and a
    /// predicate the kernel cannot decide is dropped with the kernel's reason
    /// attached. A shear is therefore not allowed to keep `LengthIs` or
    /// `RightAngle` by inertia, and a permutation that would send two points to
    /// one name is refused before anything is built.
    pub fn checked(&self, graph: &SceneGraph) -> anyhow::Result<TransformReport> {
        let affine = self.affine(graph)?;
        let rename = self.rename_map(graph)?;
        let determinant = affine.determinant()?;
        anyhow::ensure!(
            !determinant.is_zero(),
            GeometryError::NoRationalSolution(format!(
                "{} has a zero determinant, so it is not a change of presentation at all",
                self.describe()
            ))
        );
        // A shear turns circles into ellipses. The kernel names a circle by a
        // centre and a squared radius, and an ellipse has neither, so a
        // sheared circle is refused here and named in the report -- rather than
        // carried with a radius the figure no longer has.
        let shears_circles = !affine.is_similarity()? && !graph.circles.is_empty();

        let mut points: Vec<KPoint> = Vec::new();
        for point in &graph.points {
            let (x, y) = graph.coords(&point.name)?;
            let (nx, ny) = affine.map(x, y)?;
            let name = rename
                .get(&point.name)
                .cloned()
                .unwrap_or_else(|| point.name.clone());
            points.push(KPoint {
                name,
                x: Frac::from_q(nx),
                y: Frac::from_q(ny),
            });
        }
        // Two transformed points at one location is a broken scene, and the
        // only way to find out is to look: every transform here is injective by
        // construction, so a collision means the affine part is wrong.
        for (index, first) in points.iter().enumerate() {
            for second in &points[index + 1..] {
                anyhow::ensure!(
                    !(first.x == second.x && first.y == second.y),
                    GeometryError::CoincidentPoints {
                        a: first.name.clone(),
                        b: second.name.clone(),
                    }
                );
            }
        }

        let mut circles: Vec<KCircle> = Vec::new();
        if !shears_circles {
            let scale = affine.squared_length_scale()?;
            for circle in &graph.circles {
                let center = rename
                    .get(&circle.center)
                    .cloned()
                    .unwrap_or_else(|| circle.center.clone());
                let radius = circle.radius_sq.to_q()?.mul(&scale)?;
                circles.push(KCircle {
                    name: circle.name.clone(),
                    center,
                    radius_sq: Frac::from_q(radius),
                });
            }
        }

        let coords: Vec<(String, Q, Q)> = points
            .iter()
            .map(|p| Ok((p.name.clone(), p.x.to_q()?, p.y.to_q()?)))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut kept: Vec<Fact> = Vec::new();
        let mut retargeted: Vec<Constraint> = Vec::new();
        let mut dropped: Vec<DroppedFact> = Vec::new();
        for fact in &graph.facts {
            let remapped = remap_constraint(&fact.constraint, &rename, &affine)?;
            let was_retargeted = remapped != fact.constraint;
            if shears_circles && constraint_mentions_a_circle(&remapped) {
                let class = classify(&remapped);
                dropped.push(DroppedFact {
                    constraint: remapped,
                    class,
                    reason: DropReason::CircleBecomesEllipse,
                });
                continue;
            }
            let class = classify(&remapped);
            match constraint_holds_in(&circles, &coords, &remapped) {
                Ok(true) => {
                    if was_retargeted {
                        retargeted.push(remapped.clone());
                    }
                    kept.push(Fact {
                        constraint: remapped,
                        ..fact.clone()
                    });
                }
                Ok(false) => {
                    dropped.push(DroppedFact {
                        constraint: remapped,
                        class,
                        reason: DropReason::Refuted,
                    });
                }
                Err(why) => {
                    dropped.push(DroppedFact {
                        constraint: remapped,
                        class,
                        reason: DropReason::Undecidable(why.to_string()),
                    });
                }
            }
        }

        // `depends_on` and a derived fact's `inputs` name *facts* by their
        // description, so a permutation moves those names too. A dependency
        // whose fact did not survive is dropped from the list rather than left
        // pointing at something the graph no longer states -- a certificate
        // naming a missing fact is worse than a certificate naming fewer.
        let keys: BTreeMap<String, String> = kept
            .iter()
            .zip(graph.facts.iter())
            .map(|(new, old)| (old.key(), new.key()))
            .collect();
        for fact in kept.iter_mut() {
            fact.depends_on = fact
                .depends_on
                .iter()
                .filter_map(|dep| keys.get(dep).cloned())
                .collect();
            if let Provenance::Derived { inputs, .. } = &mut fact.provenance {
                *inputs = inputs
                    .iter()
                    .filter_map(|dep| keys.get(dep).cloned())
                    .collect();
            }
        }

        let mut scene = SceneGraph {
            points,
            facts: kept,
            circles,
            geometry: graph.geometry.clone(),
            not_established: graph.not_established.clone(),
        };
        for loss in &dropped {
            scene.not_established.push(format!(
                "refused: {} is not carried by {} -- {}",
                loss.constraint.describe(),
                self.describe(),
                loss.reason.describe()
            ));
        }

        let mut classes: Vec<ClassTally> = PredicateClass::ALL
            .iter()
            .map(|class| ClassTally {
                class: *class,
                ..ClassTally::empty()
            })
            .collect();
        for fact in &graph.facts {
            classes[classify(&fact.constraint).index()].examined += 1;
        }
        for fact in &scene.facts {
            classes[classify(&fact.constraint).index()].kept += 1;
        }
        for loss in &dropped {
            classes[loss.class.index()].dropped += 1;
        }

        let kept_constraints: Vec<Constraint> =
            scene.facts.iter().map(|f| f.constraint.clone()).collect();
        let confidence_after = scene_confidence(&scene);
        Ok(TransformReport {
            transform: self.clone(),
            presentation: self.describe(),
            graph: scene,
            kept: kept_constraints,
            retargeted,
            dropped,
            confidence_before: scene_confidence(graph),
            confidence_after,
            classes,
        })
    }
}

// ------------------------------------------------------------ the report --

/// Why a fact did not survive a transform. The variants are the three honest
/// outcomes of re-deciding a predicate, and the distinction between them is the
/// whole point: "the figure now says this is false" and "the kernel has no
/// semantics for this any more" are different failures, and a report that
/// conflated them would be able to hide a gap behind a refutation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum DropReason {
    /// The exact predicate came back *false* of the transformed figure. This is
    /// the expected outcome for a length or angle under a shear, and the
    /// outcome that must never happen for a similarity.
    Refuted,
    /// The exact predicate could not be decided -- a point the scene does not
    /// declare, a degeneracy the predicate refuses. The kernel's own message
    /// is carried along rather than replaced, because the reason it refused is
    /// the useful part.
    Undecidable(String),
    /// A circle was in the scene and the transform is not a similarity, so the
    /// image is an ellipse. The kernel names circles by a centre and a squared
    /// radius and has no word for an ellipse, so the claim is dropped and named
    /// rather than carried with a radius the figure no longer has.
    CircleBecomesEllipse,
}

impl DropReason {
    /// A one-line description, for the assumption ledger a dropped fact is
    /// written onto.
    pub fn describe(&self) -> String {
        match self {
            Self::Refuted => "the exact predicate rejects it of the transformed figure".to_string(),
            Self::Undecidable(why) => format!("the kernel cannot decide it any more: {why}"),
            Self::CircleBecomesEllipse => {
                "a non-similarity maps the circle to an ellipse, which the kernel cannot name"
                    .to_string()
            }
        }
    }
}

impl fmt::Display for DropReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.describe())
    }
}

/// A fact a transform did not carry, with its class and the reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DroppedFact {
    /// The constraint as it was written in the transformed figure -- the
    /// predicate that was re-decided and did not survive.
    pub constraint: Constraint,
    /// Which family the lost claim belonged to, so a report reader can see at a
    /// glance that only metric claims were lost.
    pub class: PredicateClass,
    /// Why it was lost.
    pub reason: DropReason,
}

/// What one transform did to one predicate class.
///
/// `examined + dropped == examined` is trivially true and `kept + dropped ==
/// examined` is the check that matters: it says every fact of this class was
/// either carried or accounted for. A class the scene never mentions has three
/// zeros, which means "nothing to test here" rather than "nothing went wrong",
/// and the report's verdict is written to read those two cases differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassTally {
    /// The class.
    pub class: PredicateClass,
    /// Facts of this class in the source scene.
    pub examined: usize,
    /// Facts of this class carried into the transformed scene.
    pub kept: usize,
    /// Facts of this class lost, with a reason each.
    pub dropped: usize,
}

impl ClassTally {
    fn empty() -> Self {
        Self {
            class: PredicateClass::Incidence,
            examined: 0,
            kept: 0,
            dropped: 0,
        }
    }

    /// Facts of this class the scene says nothing about -- the "nothing to
    /// test" cell, as opposed to the "tested and all survived" one.
    pub fn is_absent(&self) -> bool {
        self.examined == 0
    }
}

/// Everything one transform did to one scene, in a form a test or an audit can
/// read without re-deriving anything.
///
/// The fields answer the questions the module doc promised to answer: what
/// survived, what did not and why, what happened to the scene's confidence, and
/// how the losses break down by predicate class. Nothing in here is computed
/// from the transform's own opinion of itself -- every count comes from the
/// kernel re-deciding the predicate against transformed coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformReport {
    /// The transform applied.
    pub transform: Transform,
    /// A one-line description of it, for a table or a log line.
    pub presentation: String,
    /// The transformed scene: points moved exactly, circles rescaled, facts
    /// filtered to those that still hold.
    pub graph: SceneGraph,
    /// The constraints the transformed scene states.
    pub kept: Vec<Constraint>,
    /// Constraints whose *numeric payload* the transform had to rewrite to
    /// stay true -- a squared length multiplied by `s^2`, a radius rescaled.
    /// These are kept, not dropped, but they are listed because a reader
    /// deserves to know which facts were re-targeted rather than merely
    /// re-checked.
    pub retargeted: Vec<Constraint>,
    /// The facts that did not survive, each with its reason.
    pub dropped: Vec<DroppedFact>,
    /// The scene's confidence before the transform, and after it.
    pub confidence_before: f64,
    pub confidence_after: f64,
    /// The per-predicate-class breakdown.
    pub classes: Vec<ClassTally>,
}

impl TransformReport {
    /// The tally for one class.
    pub fn tally(&self, class: PredicateClass) -> Option<&ClassTally> {
        self.classes.iter().find(|tally| tally.class == class)
    }

    /// The losses of one class.
    pub fn losses_of(&self, class: PredicateClass) -> Vec<&DroppedFact> {
        self.dropped
            .iter()
            .filter(|loss| loss.class == class)
            .collect()
    }

    /// Whether the transform kept every fact, which is the claim the five
    /// similarity families make and the shear does not.
    pub fn kept_everything(&self) -> bool {
        self.dropped.is_empty()
    }

    /// A one-line summary, for a log.
    pub fn summary(&self) -> String {
        format!(
            "{}: kept {}/{}, dropped {} ({}), confidence {:.2} -> {:.2}",
            self.presentation,
            self.kept.len(),
            self.kept.len() + self.dropped.len(),
            self.dropped.len(),
            self.dropped
                .iter()
                .map(|loss| format!("{} {}", loss.class, loss.reason.describe()))
                .collect::<Vec<_>>()
                .join("; "),
            self.confidence_before,
            self.confidence_after
        )
    }
}

/// The scene's confidence: the weakest fact in it, or `1.0` for a scene with no
/// facts at all.
///
/// Audit fix 10 is the reason this is a minimum and not a mean. A conclusion
/// resting on a fact a diagram grader called `0.83` is worth `0.83`, and
/// averaging that against a pile of certainties is a way of rounding the
/// weakness away. An empty ledger is `1.0` because nothing has been doubted,
/// not because something has been proved.
pub fn scene_confidence(graph: &SceneGraph) -> f64 {
    graph
        .facts
        .iter()
        .map(|fact| fact.confidence)
        .fold(1.0f64, |weakest, confidence| weakest.min(confidence))
}

// --------------------------------------------------------- name remapping --

/// The new name of a point, under a (possibly empty) rename map.
fn renamed(name: &str, rename: &BTreeMap<String, String>) -> String {
    rename
        .get(name)
        .cloned()
        .unwrap_or_else(|| name.to_string())
}

/// A segment with its endpoints renamed.
fn rename_segment(seg: &Segment, rename: &BTreeMap<String, String>) -> Segment {
    Segment {
        from: renamed(&seg.from, rename),
        to: renamed(&seg.to, rename),
    }
}

/// An angle with its three points renamed.
fn rename_angle(angle: &Angle3, rename: &BTreeMap<String, String>) -> Angle3 {
    Angle3 {
        at: renamed(&angle.at, rename),
        from: renamed(&angle.from, rename),
        to: renamed(&angle.to, rename),
    }
}

/// A triangle with its three vertices renamed.
fn rename_triangle(tri: &Tri3, rename: &BTreeMap<String, String>) -> Tri3 {
    Tri3 {
        a: renamed(&tri.a, rename),
        b: renamed(&tri.b, rename),
        c: renamed(&tri.c, rename),
    }
}

/// Whether a predicate names a circle object rather than only points. Used to
/// decide whether a non-similarity has silently turned a circle into an
/// ellipse.
fn constraint_mentions_a_circle(constraint: &Constraint) -> bool {
    matches!(
        constraint,
        Constraint::Circle { .. } | Constraint::OnCircle { .. } | Constraint::Diameter { .. }
    )
}

/// Carry a predicate across a transform: rename its points, and rewrite the
/// numeric payloads that a change of scale has to rewrite.
///
/// The two jobs are separate and both are necessary. Renaming is what a
/// permutation does. Rescaling is what a similarity does to a *stated*
/// quantity -- a `LengthIs` of `16` under a scaling by `3/2` must be restated
/// as `36` or it will be re-decided as false, and the honest reason it is
/// false is that the transform rewrote the figure, not that the claim was
/// wrong. Nothing else is touched: ratios, angles and areas are stated in
/// scale-free terms and are re-decided as they stand, which is precisely how a
/// right angle is found to survive a rotation and fail a shear.
fn remap_constraint(
    constraint: &Constraint,
    rename: &BTreeMap<String, String>,
    affine: &Affine,
) -> anyhow::Result<Constraint> {
    // The scale a *squared* length is restated by -- and only a similarity has
    // one. This is the sharpest line in the module, and getting it wrong is
    // the difference between a shear that refutes length claims and a shear
    // that quietly re-states them.
    //
    // A similarity stretches every segment by the same factor, so multiplying
    // a stated square by `s^2` is a faithful restatement. A shear does not: it
    // leaves a segment along the x axis alone and stretches a perpendicular one
    // by `sqrt(1 + k^2)`, so *no* single factor is right. Multiplying by
    // `1 + k^2` would be fabrication -- and worse, it would be fabrication that
    // happens to be *right* for some segments and wrong for others, so a
    // sheared figure would come back with a plausible-looking length claim
    // that silently refers to a different measurement. So a non-similarity
    // restates nothing: the payload is carried verbatim, the exact predicate
    // re-decides it, and the claim is refuted if it no longer holds.
    let scale = if affine.is_similarity()? {
        Some(affine.squared_length_scale()?)
    } else {
        None
    };
    let square = |value: &Frac| -> anyhow::Result<Frac> {
        Ok(match &scale {
            Some(factor) => Frac::from_q(value.to_q()?.mul(factor)?),
            None => *value,
        })
    };
    let name = |point: &str| renamed(point, rename);
    Ok(match constraint {
        Constraint::Collinear { a, b, c } => Constraint::Collinear {
            a: name(a),
            b: name(b),
            c: name(c),
        },
        Constraint::Parallel { first, second } => Constraint::Parallel {
            first: rename_segment(first, rename),
            second: rename_segment(second, rename),
        },
        Constraint::Perpendicular { first, second } => Constraint::Perpendicular {
            first: rename_segment(first, rename),
            second: rename_segment(second, rename),
        },
        Constraint::EqualLength { first, second } => Constraint::EqualLength {
            first: rename_segment(first, rename),
            second: rename_segment(second, rename),
        },
        // The midpoint is the ratio `1 : 1`, which no change of scale touches:
        // this arm is deliberately a rename and nothing else.
        Constraint::MidpointOf { p, a, b } => Constraint::MidpointOf {
            p: name(p),
            a: name(a),
            b: name(b),
        },
        Constraint::Distinct { a, b } => Constraint::Distinct {
            a: name(a),
            b: name(b),
        },
        Constraint::NonCollinear { a, b, c } => Constraint::NonCollinear {
            a: name(a),
            b: name(b),
            c: name(c),
        },
        Constraint::Triangle { a, b, c } => Constraint::Triangle {
            a: name(a),
            b: name(b),
            c: name(c),
        },
        Constraint::Between { a, m, b } => Constraint::Between {
            a: name(a),
            m: name(m),
            b: name(b),
        },
        Constraint::RatioOf { p, a, b, num, den } => Constraint::RatioOf {
            p: name(p),
            a: name(a),
            b: name(b),
            num: *num,
            den: *den,
        },
        // The one length arm that states a number, so the one that has to be
        // restated: `LengthIs{AB} = 16` under a scaling by `3/2` becomes
        // `LengthIs{A'B'} = 36`, and 36 is exactly `(3/2)^2 * 16`.
        Constraint::LengthIs { seg, square: value } => Constraint::LengthIs {
            seg: rename_segment(seg, rename),
            square: square(value)?,
        },
        // A length *ratio* between two segments. Both lengths pick up the same
        // factor, so the ratio is invariant and the payload is left alone --
        // rewriting it would be the module's one temptation to be wrong.
        Constraint::ScaleLength {
            first,
            second,
            num,
            den,
        } => Constraint::ScaleLength {
            first: rename_segment(first, rename),
            second: rename_segment(second, rename),
            num: *num,
            den: *den,
        },
        Constraint::AngleEqual { first, second } => Constraint::AngleEqual {
            first: rename_angle(first, rename),
            second: rename_angle(second, rename),
        },
        Constraint::RightAngle { at } => Constraint::RightAngle {
            at: rename_angle(at, rename),
        },
        // A cosine is invariant under every similarity and is *not* a number a
        // scale multiplies, so it is carried as it stands and re-decided. A
        // shear therefore refutes it, which is the point.
        Constraint::AngleIs { at, cos } => Constraint::AngleIs {
            at: rename_angle(at, rename),
            cos: *cos,
        },
        Constraint::AreaEqual { first, second } => Constraint::AreaEqual {
            first: rename_triangle(first, rename),
            second: rename_triangle(second, rename),
        },
        Constraint::Congruent { first, second } => Constraint::Congruent {
            first: rename_triangle(first, rename),
            second: rename_triangle(second, rename),
        },
        // A circle's radius is a squared length, so it is restated by the same
        // factor; the centre is a point and follows the rename.
        Constraint::Circle {
            name: circle,
            center,
            radius_sq,
        } => Constraint::Circle {
            name: circle.clone(),
            center: name(center),
            radius_sq: square(radius_sq)?,
        },
        Constraint::OnCircle { p, circle } => Constraint::OnCircle {
            p: name(p),
            circle: circle.clone(),
        },
        Constraint::Diameter { circle, a, b } => Constraint::Diameter {
            circle: circle.clone(),
            a: name(a),
            b: name(b),
        },
    })
}

// ------------------------------------------------------- the trial families --

/// A deterministic family of trial transforms for one family row of the
/// matrix.
///
/// The trials are a *finite catalogue*, not a random sample, and that is
/// deliberate. A randomized augmentation report is a report about the seed as
/// much as about the transform; a catalogue of exactly-decidable transforms
/// (`cos^2 + sin^2 == 1`, positive rational scale) means the same invocation
/// always audits the same claims, so a regression shows up as a changed count
/// rather than as a mood.
fn rotation_trials() -> anyhow::Result<Vec<Transform>> {
    // `(cos numerator, cos denominator, sin numerator, sin denominator)`, all
    // of them exact and all of them on the unit circle: the quarter turn, the
    // half turn, the 3-4-5 turn, a 5-12-13 turn and a 3-4-5 turn the other
    // way. A 60-degree turn is absent on purpose -- its sine is `sqrt(3)/2` and
    // there is no exact rational to put here.
    let candidates: [(i128, i128, i128, i128); 5] = [
        (0, 1, 1, 1),
        (-1, 1, 0, 1),
        (3, 5, 4, 5),
        (5, 13, 12, 13),
        (4, 5, -3, 5),
    ];
    let mut out = Vec::new();
    for (cos, sin_c, sin_n, sin_d) in candidates {
        out.push(Transform::rotation(
            Q {
                num: cos,
                den: sin_c,
            },
            Q {
                num: sin_n,
                den: sin_d,
            },
        )?);
    }
    Ok(out)
}

/// Reflections: the two coordinate axes plus, when the scene offers two
/// distinct points, the line through them. A reflection in a line the figure
/// does not contain is a legal but useless trial, so the catalogue is built
/// from what the scene can name.
fn reflection_trials(graph: &SceneGraph) -> Vec<Transform> {
    let mut out = vec![
        Transform::reflection_over_x_axis(),
        Transform::reflection_over_y_axis(),
    ];
    if let (Some(a), Some(b)) = (graph.points.first(), graph.points.get(1)) {
        if (a.x, a.y) != (b.x, b.y) {
            out.push(Transform::reflection_over_points(&a.name, &b.name));
        }
    }
    out
}

/// Translations by exact rational vectors, cycling through a small catalogue.
fn translation_trials() -> Vec<Transform> {
    let vectors: [(i64, i64); 6] = [(1, 0), (0, 1), (-1, 0), (0, -1), (2, 3), (-5, 1)];
    vectors
        .iter()
        .map(|(dx, dy)| Transform::translation(Q::from_int(*dx), Q::from_int(*dy)))
        .collect()
}

/// Uniform scalings by exact positive rationals. The catalogue includes a
/// fraction below one and one above, because a scaling that only ever enlarges
/// would not notice a radius being rescaled the wrong way round.
fn scaling_trials() -> anyhow::Result<Vec<Transform>> {
    let factors: [(i128, i128); 5] = [(2, 1), (1, 2), (3, 2), (2, 3), (5, 4)];
    let mut out = Vec::new();
    for (num, den) in factors {
        out.push(Transform::scaling(
            Q { num, den: 1 },
            Q { num: den, den: 1 },
        )?);
    }
    Ok(out)
}

/// Vertex permutations: the cyclic relabellings of the scene's own point names,
/// plus the reversal where the count allows it.
///
/// A cyclic relabelling is the interesting permutation, not an arbitrary one:
/// `A -> B -> C -> A` re-presents the *same* figure with every name moved, and
/// a fact that only survives because of what its names happen to be is exposed
/// immediately, because none of the names stay put.
fn permutation_trials(graph: &SceneGraph) -> Vec<Transform> {
    let names: Vec<String> = graph.points.iter().map(|p| p.name.clone()).collect();
    if names.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for shift in 1..names.len() {
        let rename = names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), names[(index + shift) % names.len()].clone()))
            .collect();
        out.push(Transform::permutation(rename));
    }
    out
}

/// Shears by exact rationals, mixing signs and sizes. A shear with `k = 0` is
/// excluded: it is the identity, and including it would put a row of "nothing
/// broke" into the matrix under the family whose whole job is to break things.
fn shear_trials() -> Vec<Transform> {
    let factors: [(i128, i128); 5] = [(1, 1), (-1, 1), (2, 1), (1, 2), (3, 4)];
    factors
        .iter()
        .map(|(num, den)| Transform::shear(Q::new(*num, *den).unwrap_or(Q::ONE)))
        .collect()
}

/// The trial catalogue for every family, in report order.
fn trial_catalogue(graph: &SceneGraph) -> anyhow::Result<Vec<(TransformFamily, Vec<Transform>)>> {
    Ok(vec![
        (TransformFamily::Rotation, rotation_trials()?),
        (TransformFamily::Reflection, reflection_trials(graph)),
        (TransformFamily::Translation, translation_trials()),
        (TransformFamily::Scaling, scaling_trials()?),
        (TransformFamily::Permutation, permutation_trials(graph)),
        (TransformFamily::Shear, shear_trials()),
    ])
}

// -------------------------------------------------------- the invariance matrix --

/// One cell of the confusion matrix: what one transform family did to one
/// predicate class, over every trial of that family.
///
/// The three counts are the three answers the doc comment promised -- *kept*,
/// *lost*, and *nothing to test* -- and the third is the one a hand-wave always
/// leaves out. A scene with no `Congruent` fact produces a cell of three zeros
/// for congruence, and reading that as "the shear preserved congruence" would
/// be reading a number that was never collected. [`ClassCell::is_vacuous`]
/// exists so the verdict can say the difference out loud.
///
/// Totals are over *facts*, not over transforms: a class with two facts
/// examined under five trials has `examined == 2` (what the scene said) and
/// `trials` (how many times it was put through the family), with
/// `kept + lost == examined * trials`. That identity is what makes the matrix
/// checkable rather than decorative, and [`InvarianceReport::verdict`] checks
/// it before it checks anything else.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassCell {
    /// The transform family.
    pub transform: TransformFamily,
    /// The predicate class.
    pub class: PredicateClass,
    /// How many trials of the family were run.
    pub trials: usize,
    /// How many facts of this class the scene stated.
    pub examined: usize,
    /// How many (fact, trial) pairs survived.
    pub kept: usize,
    /// How many (fact, trial) pairs were lost, with a reason each in the
    /// report.
    pub lost: usize,
    /// Whether the family kept *every* fact of this class, as it must if the
    /// family is a similarity.
    pub preserved_all: bool,
    /// Whether the family lost *every* fact of this class, as the shear must if
    /// the class is one a shear destroys.
    pub refuted_all: bool,
    /// Whether the family lost *at least one* fact of this class -- the weaker
    /// claim the shear's verdict actually rests on, and the honest one.
    ///
    /// The two are not the same, and the difference is worth stating because
    /// it is easy to over-claim. A shear stretches a segment perpendicular to
    /// its axis by `sqrt(1 + k^2)` and leaves a segment along the axis exactly
    /// where it was, so a *particular* metric claim whose segments are parallel
    /// to the shear axis can survive by accident. That is a fact about that
    /// figure, not a fact about the family, and the right way to record it is a
    /// `kept` count beside a `lost` one -- not a pass/fail flag that would
    /// either wave the accident through or condemn the family for it.
    pub refuted_any: bool,
}

impl ClassCell {
    /// Whether the scene said nothing of this class, so the cell reports no
    /// evidence in either direction.
    pub fn is_vacuous(&self) -> bool {
        self.examined == 0
    }

    /// Whether the counts add up: `kept + lost == examined * trials`. A cell
    /// that fails this is a bug in the bookkeeping, and the verdict says so
    /// before it says anything about the geometry.
    pub fn is_consistent(&self) -> bool {
        self.kept + self.lost == self.examined * self.trials
    }
}

/// The full invariance report: one cell per (family, class), plus the raw
/// per-trial reports so a reader can see which individual transform did what.
///
/// The cells are the summary and the trials are the evidence. Keeping both is
/// the difference between a claim and an audit: "the shear refuted every angle"
/// is only meaningful if the angle that survived was not an accident of the one
/// `k` that was tried, and the trial list is what lets a reader check that.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InvarianceReport {
    /// The scene the report is about.
    pub graph: SceneGraph,
    /// The per-trial reports, in catalogue order.
    pub trials: Vec<TransformReport>,
    /// The matrix, one cell per (family, class), in [`TransformFamily::ALL`]
    /// then [`PredicateClass::ALL`] order.
    pub cells: Vec<ClassCell>,
    /// Refusals met while building the catalogue, with the transform that caused
    /// them. A trial that could not be built is reported, never skipped in
    /// silence.
    pub refusals: Vec<String>,
}

impl InvarianceReport {
    /// The cell for one (family, class) pair.
    pub fn cell(&self, family: TransformFamily, class: PredicateClass) -> Option<&ClassCell> {
        self.cells
            .iter()
            .find(|cell| cell.transform == family && cell.class == class)
    }

    /// The cells of one family, in class order.
    pub fn row(&self, family: TransformFamily) -> Vec<&ClassCell> {
        self.cells
            .iter()
            .filter(|cell| cell.transform == family)
            .collect()
    }

    /// The classes the shear is expected to break: the metric ones. Stated as
    /// a function so the verdict and the tests agree on what "the shear broke
    /// the metric" means -- and as a *class* list rather than a per-fact claim,
    /// because whether a given length claim survives a shear depends on the
    /// direction of its segments, not on the shear.
    pub fn shear_destroys(class: PredicateClass) -> bool {
        matches!(
            class,
            PredicateClass::Length | PredicateClass::Angle | PredicateClass::Congruence
        )
    }

    /// Whether the report is internally consistent: every cell's counts add up.
    pub fn is_consistent(&self) -> bool {
        self.cells.iter().all(ClassCell::is_consistent)
    }

    /// The verdict: whether each family behaved exactly as it claimed, in
    /// plain sentences a reader can disagree with.
    ///
    /// The claims being checked, stated so that "passed" means something:
    ///
    /// - each of the five similarity families must keep **every** fact of
    ///   **every** class. A single lost incidence under a quarter turn is a
    ///   failure of that family, and the verdict names the class;
    /// - the shear must keep every incidence, parallelism, ratio and area
    ///   fact, and must lose every length, angle and congruence fact the scene
    ///   actually has. A shear that kept a right angle is a shear whose
    ///   augmentation teaches the model to ignore the drawing, and the verdict
    ///   says so by name.
    pub fn verdict(&self) -> String {
        let mut lines = Vec::new();
        if !self.is_consistent() {
            lines.push(
                "the report does not add up: some cell's kept and lost counts do not match the \
                 facts examined, so nothing below can be trusted"
                    .to_string(),
            );
            return lines.join("\n");
        }
        lines.push(format!(
            "invariance report over {} trials of {} families on a scene of {} points and {} facts",
            self.trials.len(),
            TransformFamily::ALL.len(),
            self.graph.points.len(),
            self.graph.facts.len()
        ));
        for family in TransformFamily::ALL {
            let row = self.row(family);
            let failures: Vec<String> = row
                .iter()
                .filter(|cell| {
                    if cell.is_vacuous() {
                        return false;
                    }
                    if family.is_exact() {
                        !cell.preserved_all
                    } else if Self::shear_destroys(cell.class) {
                        // The shear must destroy the metric classes -- at least
                        // one fact of each, which is what proves it can tell a
                        // sheared figure from an unstretched one. `refuted_all`
                        // is *not* required, and the reason is stated on the
                        // field: a metric claim whose segments run along the
                        // shear's axis survives by accident, and failing the
                        // family over that accident would make the check a
                        // statement about the fixture rather than about the
                        // transform.
                        !cell.refuted_any
                    } else {
                        // ...and touch nothing that is not metric.
                        !cell.preserved_all
                    }
                })
                .map(|cell| format!("{} ({})", cell.class, describe_cell(cell)))
                .collect();
            let observed: Vec<String> = row
                .iter()
                .filter(|cell| !cell.is_vacuous())
                .map(|cell| format!("{} {}/{}", cell.class, cell.kept, cell.kept + cell.lost))
                .collect();
            lines.push(format!(
                "{:<12} {:<52} {}",
                family.to_string(),
                if observed.is_empty() {
                    "the scene states nothing to test".to_string()
                } else {
                    observed.join(", ")
                },
                if failures.is_empty() {
                    "as claimed".to_string()
                } else {
                    format!("NOT as claimed: {}", failures.join(", "))
                }
            ));
        }
        let discrimination = self.discrimination_sentence();
        if let Some(sentence) = discrimination {
            lines.push(sentence);
        }
        if !self.refusals.is_empty() {
            lines.push(format!("refusals met: {}", self.refusals.join("; ")));
        }
        lines.join("\n")
    }

    /// The sentence the module doc promised: whether the shear actually
    /// discriminated, which is the difference between an augmentation that
    /// forces a model to re-read the figure and one that lets it answer from a
    /// memorized template.
    ///
    /// `None` when the scene states no metric fact at all -- there is nothing
    /// for the shear to refute, and claiming the shear "taught the lesson" off
    /// the back of a figure with no lengths in it would be the exact kind of
    /// over-claim this module exists to stop.
    fn discrimination_sentence(&self) -> Option<String> {
        let row = self.row(TransformFamily::Shear);
        let metric: Vec<&ClassCell> = row
            .iter()
            .copied()
            .filter(|cell| Self::shear_destroys(cell.class) && !cell.is_vacuous())
            .collect();
        if metric.is_empty() {
            return None;
        }
        let total = metric.iter().map(|cell| cell.examined).sum::<usize>();
        let lost = metric.iter().map(|cell| cell.lost).sum::<usize>();
        let expected = metric
            .iter()
            .map(|cell| cell.examined * cell.trials)
            .sum::<usize>();
        if lost == expected && expected > 0 {
            Some(format!(
                "the shear refuted all {total} metric facts of this scene, so a model answering a \
                 sheared figure with the unstretched answer is caught"
            ))
        } else {
            Some(format!(
                "the shear refuted {lost} of the {expected} metric-fact checks on this scene: a model \
                 could answer some sheared figures from a memorized template, and the report says so"
            ))
        }
    }
}

/// A cell's headline number, for a verdict line: `kept/total`, or the word
/// `vacuous` when the scene said nothing of that class.
fn describe_cell(cell: &ClassCell) -> String {
    if cell.is_vacuous() {
        "vacuous".to_string()
    } else {
        format!("{}/{} kept", cell.kept, cell.kept + cell.lost)
    }
}

/// Run every family of transforms against a scene and report, predicate class
/// by predicate class, whether each family behaved as it claimed.
///
/// `trials` is a cap on how many transforms per family are run, drawn in
/// catalogue order; asking for more than the catalogue holds repeats nothing
/// and simply runs the whole catalogue, because repeating a trial adds no
/// evidence and inventing an irrational one would add a number the kernel
/// cannot decide.
///
/// The shear is included on purpose. An invariance report that only checked the
/// families it expected to pass would be a report about nothing, and the whole
/// reason the shear is in this library is that it is the family that must
/// *fail* to preserve the metric classes. A scene with circles has its circles
/// dropped by the shear and the reason recorded, which is a different failure
/// from a refuted length and is tallied as one.
pub fn invariance_report(graph: &SceneGraph, trials: usize) -> anyhow::Result<InvarianceReport> {
    let catalogue = trial_catalogue(graph)?;
    let mut reports: Vec<TransformReport> = Vec::new();
    let mut refusals: Vec<String> = Vec::new();
    for (family, transforms) in catalogue {
        for transform in transforms.into_iter().take(trials.max(1)) {
            match transform.apply_reported(graph) {
                Ok(report) => reports.push(report),
                Err(why) => refusals.push(format!("{family}: {why}")),
            }
        }
    }
    let mut cells: Vec<ClassCell> = Vec::new();
    for family in TransformFamily::ALL {
        for class in PredicateClass::ALL {
            let row: Vec<&TransformReport> = reports
                .iter()
                .filter(|report| report.transform.family() == family)
                .collect();
            let examined = row
                .first()
                .map(|report| report.tally(class).map_or(0, |tally| tally.examined))
                .unwrap_or(0);
            let kept: usize = row
                .iter()
                .map(|report| report.tally(class).map_or(0, |tally| tally.kept))
                .sum();
            let lost: usize = row
                .iter()
                .map(|report| report.tally(class).map_or(0, |tally| tally.dropped))
                .sum();
            let total = examined * row.len();
            // A family with no trials at all proves nothing, and must not read
            // as a pass: `kept == total == 0` would otherwise score a vacuous
            // "kept everything" for a family the catalogue never ran.
            let ran = !row.is_empty();
            cells.push(ClassCell {
                transform: family,
                class,
                trials: row.len(),
                examined,
                kept,
                lost,
                preserved_all: ran && examined > 0 && kept == total,
                refuted_all: ran && examined > 0 && lost == total,
                refuted_any: lost > 0,
            });
        }
    }
    Ok(InvarianceReport {
        graph: graph.clone(),
        trials: reports,
        cells,
        refusals,
    })
}

// ------------------------------------------------------ augmented examples --

/// One surface presentation of a proof graph, paired with the answer it should
/// elicit.
///
/// The pairing is the training signal and the reason this type exists. A
/// transformed scene on its own says nothing about what a model should say
/// about it; a scene plus the claim that is true of it says "this figure, this
/// answer", and the same underlying proof graph arrives in many disguises. When
/// the transform broke a fact -- which is what the shear does to the metric
/// classes -- `refuted` says so, and the claim is the *transformed* one, so a
/// trainer pairing this example must not teach the original answer on a figure
/// that no longer supports it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AugmentedExample {
    /// The transform that produced this presentation.
    pub transform: Transform,
    /// A one-line description of it, for a dataset index.
    pub presentation: String,
    /// The presented figure.
    pub graph: SceneGraph,
    /// The claim to answer with: the original claim, carried across the
    /// transform (renamed, and with any stated squared length rescaled).
    pub claim: Constraint,
    /// Whether that carried claim is actually true of the presented figure, as
    /// the kernel decides it. A sheared presentation of a `LengthIs` claim
    /// carries the claim with `claim_holds == false`, and a trainer that pairs
    /// it with the original answer would be teaching the model to be wrong on
    /// purpose -- so the flag is here to make that impossible to do by
    /// accident.
    pub claim_holds: bool,
    /// The facts the transform dropped, so a trainer can see that this example
    /// is not equivalent to the original.
    pub refuted: Vec<Constraint>,
    /// The per-class tally for this example.
    pub classes: Vec<ClassTally>,
    /// The scene's confidence before and after.
    pub confidence_before: f64,
    pub confidence_after: f64,
}

impl AugmentedExample {
    /// Whether this presentation lost nothing: the same proof graph, restated,
    /// with an answer still true of it. A shear example is not one of these,
    /// and that is the whole point of being able to tell.
    pub fn is_equivalent(&self) -> bool {
        self.refuted.is_empty() && self.claim_holds
    }

    /// The class of the carried claim.
    pub fn claim_class(&self) -> PredicateClass {
        self.claim.classify()
    }
}

/// Produce `count` surface presentations of one proof graph, each paired with
/// the claim it should answer.
///
/// The generator is the module's payoff and its discipline at the same time. It
/// walks the same catalogue the invariance report audits, so every example it
/// emits is a transform whose behaviour has been *checked* rather than
/// assumed, and it records in every example which facts the presentation lost
/// so that a "different surface, same answer" claim is never made about a
/// figure that refutes the claim.
///
/// Two filters keep the output honest as training data:
///
/// - a presentation that duplicates an earlier one's *coordinates and names* is
///   skipped, because a dataset of the same picture under five labels is the
///   duplication the module doc railed against;
/// - examples are drawn across families in turn, so a generator asked for
///   four examples gives four *kinds* of presentation rather than four
///   translations of the first one.
pub fn augmentations(
    graph: &SceneGraph,
    claim: Constraint,
    count: usize,
) -> anyhow::Result<Vec<AugmentedExample>> {
    let mut out: Vec<AugmentedExample> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for (_family, transforms) in trial_catalogue(graph)? {
        for transform in transforms {
            if out.len() >= count {
                break;
            }
            let report = match transform.apply_reported(graph) {
                Ok(report) => report,
                // A transform the scene cannot take (a reflection axis through
                // coincident points, a permutation that is not a bijection) is
                // not an example and not an error: the catalogue is a
                // superset of what any one scene can accept.
                Err(_) => continue,
            };
            let rename = transform.rename_map(graph)?;
            let signature = presentation_signature(&report.graph)?;
            if seen.contains(&signature) {
                continue;
            }
            seen.push(signature);
            // The claim is carried across *this* transform, rename included: a
            // permuted example asked about `MidpointOf(B, C, A)` would be
            // training a model on a claim about points the figure no longer
            // has. And whether the carried claim still holds in the presented
            // figure is decided by the kernel, not assumed, because under a
            // shear it generally does not.
            let carried = remap_constraint(&claim, &rename, &report.transform.affine(graph)?)?;
            let claim_holds = constraint_holds_in(
                &report.graph.circles,
                &report
                    .graph
                    .points
                    .iter()
                    .map(|point| Ok((point.name.clone(), point.x.to_q()?, point.y.to_q()?)))
                    .collect::<anyhow::Result<Vec<_>>>()?,
                &carried,
            )?;
            out.push(AugmentedExample {
                transform: report.transform.clone(),
                presentation: report.presentation.clone(),
                claim: carried,
                claim_holds,
                refuted: report
                    .dropped
                    .iter()
                    .map(|loss| loss.constraint.clone())
                    .collect(),
                classes: report.classes.clone(),
                confidence_before: report.confidence_before,
                confidence_after: report.confidence_after,
                graph: report.graph,
            });
        }
    }
    Ok(out)
}

/// A presentation's identity: its points, in name order, with exact
/// coordinates.
///
/// Two presentations with the same signature are the same picture, and offering
/// both to a trainer is duplication no matter how many transforms produced
/// them. A permutation that renamed everything produces a different signature
/// (the names are part of it) and is therefore kept, which is correct: `ABC`
/// and `BCA` are genuinely different surface forms even though the figure is
/// the same.
fn presentation_signature(graph: &SceneGraph) -> anyhow::Result<String> {
    let mut points: Vec<String> = graph
        .points
        .iter()
        .map(|point| Ok(format!("{}:{},{}", point.name, point.x, point.y)))
        .collect::<anyhow::Result<Vec<_>>>()?;
    points.sort();
    let mut facts: Vec<String> = graph.facts.iter().map(|fact| fact.key()).collect();
    facts.sort();
    Ok(format!("{}|{}", points.join(";"), facts.join(";")))
}

// ------------------------------------------------------------------ tests --

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
    use crate::geomkernel::{doubled_area, graph_from_points, Fact, KCircle, KPoint, QSqrt};

    /// An exact rational, for the values a test asserts against.
    fn q(num: i128, den: i128) -> Q {
        Q::new(num, den).unwrap()
    }

    /// An exact rational as the kernel serializes it.
    fn f(num: i128, den: i128) -> Frac {
        Frac { num, den }
    }

    /// A lattice point.
    fn p(name: &str, x: i64, y: i64) -> KPoint {
        KPoint {
            name: name.to_string(),
            x: f(i128::from(x), 1),
            y: f(i128::from(y), 1),
        }
    }

    /// The segment `AB`, named the way the kernel names a line.
    fn seg(from: &str, to: &str) -> Segment {
        Segment {
            from: from.to_string(),
            to: to.to_string(),
        }
    }

    /// The triangle `ABC`.
    fn tri(a: &str, b: &str, c: &str) -> Tri3 {
        Tri3 {
            a: a.to_string(),
            b: b.to_string(),
            c: c.to_string(),
        }
    }

    /// A typed refusal. A refusal that is not a `GeometryError` is a test
    /// failure, not a shrug: this module's claim is that it refuses *by name*.
    fn refusal<T>(result: anyhow::Result<T>) -> GeometryError {
        result
            .err()
            .and_then(|why| why.downcast::<GeometryError>().ok())
            .expect("a typed GeometryError")
    }

    /// The reference figure, with every fact true of it.
    ///
    /// `A(0,0) B(4,0) C(0,4) D(2,0) E(0,2) F(4,4)`: a square's corner with the
    /// two midpoints and the opposite corner, so that incidence, parallelism,
    /// ratio, length, angle, area and congruence all have a true statement to
    /// make. The facts are added through [`SceneGraph::add_fact`] rather than
    /// through `graph_from_points`, because that constructor marks everything
    /// `given` *without checking it* -- a fixture built with it would be a pile
    /// of assertions the module then dutifully tried to preserve.
    fn triangle_scene() -> SceneGraph {
        let mut graph = graph_from_points(
            vec![
                p("A", 0, 0),
                p("B", 4, 0),
                p("C", 0, 4),
                p("D", 2, 0),
                p("E", 0, 2),
                p("F", 4, 4),
            ],
            vec![],
        );
        let stated = [
            Constraint::Distinct {
                a: "A".into(),
                b: "B".into(),
            },
            Constraint::Triangle {
                a: "A".into(),
                b: "B".into(),
                c: "C".into(),
            },
            Constraint::NonCollinear {
                a: "A".into(),
                b: "B".into(),
                c: "C".into(),
            },
            Constraint::Collinear {
                a: "A".into(),
                b: "D".into(),
                c: "B".into(),
            },
            Constraint::MidpointOf {
                p: "D".into(),
                a: "A".into(),
                b: "B".into(),
            },
            Constraint::Between {
                a: "A".into(),
                m: "D".into(),
                b: "B".into(),
            },
            Constraint::RatioOf {
                p: "E".into(),
                a: "A".into(),
                b: "C".into(),
                num: 1,
                den: 1,
            },
            Constraint::Between {
                a: "A".into(),
                m: "E".into(),
                b: "C".into(),
            },
            Constraint::Parallel {
                first: seg("A", "C"),
                second: seg("B", "F"),
            },
            Constraint::EqualLength {
                first: seg("A", "C"),
                second: seg("B", "F"),
            },
            Constraint::ScaleLength {
                first: seg("A", "C"),
                second: seg("B", "F"),
                num: 1,
                den: 1,
            },
            Constraint::LengthIs {
                seg: seg("A", "B"),
                square: f(16, 1),
            },
            // A second stated length, and the reason it is here is a fact about
            // shears worth having in a fixture. `AC` runs *across* the shear's
            // axis while `AB` runs along it, so a shear moves `AC` and leaves
            // `AB` alone: this is the length claim the shear refutes, and `AB`
            // is the one it does not. A figure with only axis-parallel lengths
            // would let a shear keep every length fact, which is true and would
            // make the metric cell of the matrix vacuous.
            Constraint::LengthIs {
                seg: seg("A", "C"),
                square: f(16, 1),
            },
            Constraint::Perpendicular {
                first: seg("A", "B"),
                second: seg("A", "C"),
            },
            Constraint::RightAngle {
                at: Angle3::new("A", "B", "C"),
            },
            Constraint::AngleIs {
                at: Angle3::new("A", "B", "C"),
                cos: QSqrt::rational(f(0, 1)),
            },
            Constraint::AngleEqual {
                first: Angle3::new("A", "B", "C"),
                second: Angle3::new("B", "A", "F"),
            },
            Constraint::AreaEqual {
                first: tri("A", "B", "C"),
                second: tri("A", "B", "F"),
            },
            Constraint::Congruent {
                first: tri("A", "B", "C"),
                second: tri("A", "B", "F"),
            },
        ];
        for constraint in stated {
            assert!(
                graph.add_fact(constraint.clone(), Provenance::Given),
                "the fixture states {} but the figure does not have it: {:?}",
                constraint.describe(),
                graph.not_established
            );
        }
        graph
    }

    // ------------------------------------------------------------- vocabulary --

    #[test]
    fn test_every_predicate_has_a_class_and_they_are_all_distinct_claims() {
        // The classification is the thing the report counts by, so the first
        // claim about it is that it is total and lands where intended.
        assert_eq!(
            Constraint::Collinear {
                a: "A".into(),
                b: "B".into(),
                c: "C".into()
            }
            .classify(),
            PredicateClass::Incidence
        );
        assert_eq!(
            Constraint::Between {
                a: "A".into(),
                m: "B".into(),
                b: "C".into()
            }
            .classify(),
            PredicateClass::Incidence
        );
        assert_eq!(
            Constraint::Parallel {
                first: seg("A", "B"),
                second: seg("C", "D")
            }
            .classify(),
            PredicateClass::Parallelism
        );
        // A midpoint is a division ratio, not a location: affine invariant.
        assert_eq!(
            Constraint::MidpointOf {
                p: "A".into(),
                a: "B".into(),
                b: "C".into()
            }
            .classify(),
            PredicateClass::Ratio
        );
        assert_eq!(
            Constraint::RatioOf {
                p: "A".into(),
                a: "B".into(),
                b: "C".into(),
                num: 1,
                den: 2
            }
            .classify(),
            PredicateClass::Ratio
        );
        assert_eq!(
            Constraint::LengthIs {
                seg: seg("A", "B"),
                square: f(1, 1)
            }
            .classify(),
            PredicateClass::Length
        );
        // A perpendicularity is a right angle, and a shear does not keep those.
        assert_eq!(
            Constraint::Perpendicular {
                first: seg("A", "B"),
                second: seg("C", "D")
            }
            .classify(),
            PredicateClass::Angle
        );
        assert_eq!(
            Constraint::RightAngle {
                at: Angle3::new("A", "B", "C")
            }
            .classify(),
            PredicateClass::Angle
        );
        assert_eq!(
            Constraint::AreaEqual {
                first: tri("A", "B", "C"),
                second: tri("D", "E", "F")
            }
            .classify(),
            PredicateClass::Area
        );
        assert_eq!(
            Constraint::Congruent {
                first: tri("A", "B", "C"),
                second: tri("D", "E", "F")
            }
            .classify(),
            PredicateClass::Congruence
        );
        // Every class has a slot in the tally vector, or a loss could not be
        // recorded at all.
        let mut seen: Vec<usize> = PredicateClass::ALL.iter().map(|c| c.index()).collect();
        seen.sort();
        assert_eq!(seen, vec![0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(PredicateClass::ALL.len(), 7);
    }

    #[test]
    fn test_a_circle_is_a_metric_object_and_says_so() {
        // The one classification that could go either way, argued explicitly:
        // `OnCircle` and `Diameter` are statements about a radius, and a shear
        // maps the circle to an ellipse, so they are losses of the metric
        // family rather than broken incidences.
        assert_eq!(
            Constraint::Circle {
                name: "w".into(),
                center: "A".into(),
                radius_sq: f(1, 1)
            }
            .classify(),
            PredicateClass::Length
        );
        assert_eq!(
            Constraint::OnCircle {
                p: "A".into(),
                circle: "w".into()
            }
            .classify(),
            PredicateClass::Length
        );
        assert_eq!(
            Constraint::Diameter {
                circle: "w".into(),
                a: "A".into(),
                b: "B".into()
            }
            .classify(),
            PredicateClass::Length
        );
        assert!(constraint_mentions_a_circle(&Constraint::OnCircle {
            p: "A".into(),
            circle: "w".into()
        }));
        assert!(!constraint_mentions_a_circle(&Constraint::Collinear {
            a: "A".into(),
            b: "B".into(),
            c: "C".into()
        }));
    }

    // ------------------------------------------------------------- rotations --

    #[test]
    fn test_quarter_turn_permutes_coordinates_exactly() {
        // `(x, y) -> (-y, x)`, in rationals. No float is consulted and none is
        // needed: `cos = 0`, `sin = 1`.
        let graph = triangle_scene();
        let turned = Transform::quarter_turn().apply(&graph).unwrap();
        let (ax, ay) = turned.coords("A").unwrap();
        let (bx, by) = turned.coords("B").unwrap();
        let (cx, cy) = turned.coords("C").unwrap();
        let (fx, fy) = turned.coords("F").unwrap();
        assert_eq!((ax, ay), (q(0, 1), q(0, 1)));
        assert_eq!((bx, by), (q(0, 1), q(4, 1)));
        assert_eq!((cx, cy), (q(-4, 1), q(0, 1)));
        // `F(4,4)` is the one point that exercises the formula: it goes to
        // `(-4, 4)`, and a matrix that negated the wrong entry would send it
        // somewhere else entirely.
        assert_eq!((fx, fy), (q(-4, 1), q(4, 1)));
        // A quarter turn is a similarity, so it must keep every fact -- and
        // `A(0,0)` sitting on the origin is not why: the far points moved too.
        assert_eq!(turned.facts.len(), graph.facts.len());
    }

    #[test]
    fn test_three_four_five_turn_is_exact_where_a_float_would_round() {
        // cos = 3/5, sin = 4/5 applied to (4, 0) gives exactly (12/5, 16/5).
        // In f64 the same computation gives 2.4000000000000004 and every
        // downstream predicate is a tolerance test rather than a decision.
        let graph = triangle_scene();
        let turned = Transform::three_four_five_turn().apply(&graph).unwrap();
        let (bx, by) = turned.coords("B").unwrap();
        assert_eq!((bx, by), (q(12, 5), q(16, 5)));
        // And `C(0,4)`, which exercises the `sin` entry of the matrix:
        // `(-4 * 4/5, 4 * 3/5) = (-16/5, 12/5)`, exact.
        let (cx, cy) = turned.coords("C").unwrap();
        assert_eq!((cx, cy), (q(-16, 5), q(12, 5)));
        // And the claim it exists to make: a rational 3-4-5 turn preserves
        // every derived fact of the figure.
        let report = Transform::three_four_five_turn()
            .apply_reported(&graph)
            .unwrap();
        assert!(report.kept_everything(), "{}", report.summary());
    }

    #[test]
    fn test_a_rotation_off_the_unit_circle_is_refused() {
        // cos = 3/5, sin = 3/5 gives 18/25, not 1. Applying it would shear the
        // figure while claiming to turn it, which is the specific dishonesty
        // the exactness discipline exists to prevent.
        let refused = refusal(Transform::rotation(q(3, 5), q(3, 5)));
        assert!(
            matches!(refused, GeometryError::NoRationalSolution(_)),
            "{refused}"
        );
        // 1/2 and 1/2 (the 60-degree turn's cosine with its own sine) is the
        // same refusal: there is no exact rational sine to pair with it.
        assert!(matches!(
            refusal(Transform::rotation(q(1, 2), q(1, 2))),
            GeometryError::NoRationalSolution(_)
        ));
        // The zero pair is refused too, rather than collapsing the figure.
        assert!(matches!(
            refusal(Transform::rotation(Q::ZERO, Q::ZERO)),
            GeometryError::NoRationalSolution(_)
        ));
        // And the exact pairs are accepted.
        assert!(Transform::rotation(Q::ZERO, Q::ONE).is_ok());
        assert!(Transform::rotation(q(3, 5), q(4, 5)).is_ok());
        assert!(Transform::rotation(q(5, 13), q(12, 13)).is_ok());
    }

    // ----------------------------------------------------------- translation --

    #[test]
    fn test_translation_preserves_every_fact_exactly() {
        // A translation is the strongest invariance claim in the library: it
        // moves every point and changes not one predicate, because every
        // predicate the kernel has is a statement about differences.
        let graph = triangle_scene();
        let moved = Transform::translation(q(-7, 3), q(11, 5))
            .apply(&graph)
            .unwrap();
        let report = Transform::translation(q(-7, 3), q(11, 5))
            .apply_reported(&graph)
            .unwrap();
        assert!(report.kept_everything(), "{}", report.summary());
        assert_eq!(moved.facts.len(), graph.facts.len());
        let (ax, ay) = moved.coords("A").unwrap();
        assert_eq!((ax, ay), (q(-7, 3), q(11, 5)));
        let (bx, by) = moved.coords("B").unwrap();
        assert_eq!(
            (bx, by),
            (
                q(-7, 3).add(&q(4, 1)).unwrap(),
                q(11, 5).add(&q(0, 1)).unwrap()
            )
        );
        let (fx, fy) = moved.coords("F").unwrap();
        assert_eq!(
            (fx, fy),
            (
                q(4, 1).add(&q(-7, 3)).unwrap(),
                q(4, 1).add(&q(11, 5)).unwrap()
            )
        );
    }

    // ------------------------------------------------------------ reflection --

    #[test]
    fn test_reflection_over_the_x_axis_flips_orientation_and_keeps_every_fact() {
        let graph = triangle_scene();
        let report = Transform::reflection_over_x_axis()
            .apply_reported(&graph)
            .unwrap();
        // A reflection is a similarity, so every fact survives...
        assert!(report.kept_everything(), "{}", report.summary());
        let flipped = report.graph;
        let (bx, by) = flipped.coords("B").unwrap();
        assert_eq!((bx, by), (q(4, 1), q(0, 1)));
        let (_, cy) = flipped.coords("C").unwrap();
        assert_eq!(cy, q(-4, 1));
        // ...but the orientation is reversed, and that is a *sign* the kernel
        // keeps exactly: the doubled area of ABC is 16 before and -16 after,
        // and a reflection is the one family here that has determinant -1.
        let before = doubled_area(&coord_table(&graph), &tri("A", "B", "C")).unwrap();
        let after = doubled_area(&coord_table(&flipped), &tri("A", "B", "C")).unwrap();
        assert_eq!(before, q(16, 1));
        assert_eq!(after, q(-16, 1));
        assert!(before.less(&Q::ZERO) != after.less(&Q::ZERO));
    }

    #[test]
    fn test_reflection_over_a_line_through_two_points_fixes_them_exactly() {
        // The line through `A(0,0)` and `B(4,0)` is the x axis, so this is the
        // same figure as the axis reflection -- reached through the general
        // code path, which is the point. The two axis points must come back to
        // themselves, exactly, because a reflection fixes its own axis.
        let graph = triangle_scene();
        let report = Transform::reflection_over_points("A", "B")
            .apply_reported(&graph)
            .unwrap();
        assert!(report.kept_everything(), "{}", report.summary());
        let (ax, ay) = report.graph.coords("A").unwrap();
        let (bx, by) = report.graph.coords("B").unwrap();
        assert_eq!((ax, ay), (q(0, 1), q(0, 1)));
        assert_eq!((bx, by), (q(4, 1), q(0, 1)));
        let (cx, cy) = report.graph.coords("C").unwrap();
        assert_eq!((cx, cy), (q(0, 1), q(-4, 1)));
        // The general path agrees with the axis one on the same line, which is
        // the check that the line-through-two-points form is not a different
        // transformation wearing a longer name.
        assert_eq!(
            report.graph.coords("C").unwrap(),
            Transform::reflection_over_x_axis()
                .apply(&graph)
                .unwrap()
                .coords("C")
                .unwrap()
        );
    }

    #[test]
    fn test_a_reflection_axis_through_coincident_points_is_refused() {
        // Two names for one location have no direction, so they are not a line.
        // The kernel's `DegenerateSegment` is exactly the right refusal.
        let graph = graph_from_points(vec![p("A", 1, 1), p("B", 1, 1), p("C", 0, 0)], vec![]);
        let refused = refusal(Transform::reflection_over_points("A", "B").apply(&graph));
        assert!(
            matches!(refused, GeometryError::DegenerateSegment),
            "{refused}"
        );
    }

    // -------------------------------------------------------------- scaling --

    #[test]
    fn test_rational_scaling_keeps_every_fact_and_rescales_the_stated_length() {
        let graph = triangle_scene();
        let report = Transform::scaling(q(3, 2), Q::ONE)
            .unwrap()
            .apply_reported(&graph)
            .unwrap();
        assert!(report.kept_everything(), "{}", report.summary());
        let scaled = report.graph;
        // Every point moved by exactly 3/2.
        let (bx, by) = scaled.coords("B").unwrap();
        assert_eq!((bx, by), (q(6, 1), q(0, 1)));
        let (cx, cy) = scaled.coords("C").unwrap();
        assert_eq!((cx, cy), (q(0, 1), q(6, 1)));
        // The stated squared length was rewritten, not merely re-checked: 16
        // became 36, because (3/2)^2 * 16 = 36 exactly.
        assert!(scaled.has_fact(&Constraint::LengthIs {
            seg: seg("A", "B"),
            square: f(36, 1)
        }));
        assert!(report.retargeted.contains(&Constraint::LengthIs {
            seg: seg("A", "B"),
            square: f(36, 1)
        }));
        // And the kernel agrees: 6^2 + 0^2 = 36.
        assert!(scaled
            .holds(&Constraint::LengthIs {
                seg: seg("A", "B"),
                square: f(36, 1)
            })
            .unwrap());
        // Incidence, parallelism, ratios and angles are scale-free and were
        // carried as they stood, never rescaled.
        assert!(scaled.has_fact(&Constraint::MidpointOf {
            p: "D".into(),
            a: "A".into(),
            b: "B".into()
        }));
        assert!(scaled.has_fact(&Constraint::RightAngle {
            at: Angle3::new("A", "B", "C")
        }));
        // A shrink works the same way, which is what catches a radius rescaled
        // in only one direction.
        let shrunk = Transform::scaling(q(1, 2), Q::ONE)
            .unwrap()
            .apply_reported(&graph)
            .unwrap();
        assert!(shrunk.kept_everything(), "{}", shrunk.summary());
        assert!(shrunk.graph.has_fact(&Constraint::LengthIs {
            seg: seg("A", "B"),
            square: f(4, 1)
        }));
    }

    #[test]
    fn test_a_zero_or_negative_scale_is_refused() {
        // Zero collapses every point onto the origin: there is no figure left
        // to reason about, which is what `EmptyGeometry` says.
        let refused = refusal(Transform::scaling(Q::ZERO, Q::ONE));
        assert!(
            matches!(refused, GeometryError::EmptyGeometry(_)),
            "{refused}"
        );
        // A negative factor is a half turn wearing a scale's name, and the
        // refusal says so rather than silently mirroring the figure.
        let negative = refusal(Transform::scaling(q(-1, 2), Q::ONE));
        assert!(
            matches!(negative, GeometryError::NoRationalSolution(_)),
            "{negative}"
        );
        // A zero denominator is not a number at all.
        assert!(matches!(
            refusal(Transform::scaling(Q::ONE, Q::ZERO)),
            GeometryError::NoRationalSolution(_)
        ));
        // Positive fractions are fine.
        assert!(Transform::scaling(q(2, 3), Q::ONE).is_ok());
    }

    // ---------------------------------------------------------------- shear --

    #[test]
    fn test_shear_breaks_a_length_and_a_right_angle_while_keeping_the_structure() {
        // This is the test the module exists for. The figure has a stated
        // length, a right angle, a midpoint, a ratio, a parallelism and a
        // collinearity. The shear must destroy the first two and keep the rest,
        // and `apply` decides each one by re-running the kernel's own
        // predicate -- not by consulting a table of which classes a shear is
        // supposed to break.
        let graph = triangle_scene();
        // `k = 1/2` rather than `k = 1`, so the coordinates below are fractions
        // and a test that passes on integers is visibly passing on exact
        // rationals rather than on values that happened to divide.
        let shear = Transform::shear(q(1, 2));
        let report = shear.apply_reported(&graph).unwrap();
        // A quarter turn first, to show the contrast: it is a similarity, so it
        // loses nothing, and the two reports are the module's thesis in a pair.
        let turned = Transform::quarter_turn().apply_reported(&graph).unwrap();
        assert!(turned.kept_everything(), "{}", turned.summary());

        // Kept: the affine facts. A shear is affine, and incidence,
        // parallelism, betweenness and division ratios are affine invariants.
        for survivor in [
            Constraint::Distinct {
                a: "A".into(),
                b: "B".into(),
            },
            Constraint::Collinear {
                a: "A".into(),
                b: "D".into(),
                c: "B".into(),
            },
            Constraint::MidpointOf {
                p: "D".into(),
                a: "A".into(),
                b: "B".into(),
            },
            Constraint::Between {
                a: "A".into(),
                m: "D".into(),
                b: "B".into(),
            },
            Constraint::Parallel {
                first: seg("A", "C"),
                second: seg("B", "F"),
            },
            Constraint::RatioOf {
                p: "E".into(),
                a: "A".into(),
                b: "C".into(),
                num: 1,
                den: 1,
            },
            Constraint::AreaEqual {
                first: tri("A", "B", "C"),
                second: tri("A", "B", "F"),
            },
        ] {
            assert!(
                report.graph.has_fact(&survivor),
                "{} should survive a shear",
                survivor.describe()
            );
            assert!(
                report.tally(survivor.classify()).map_or(0, |t| t.kept) >= 1,
                "{} should be tallied as kept",
                survivor.describe()
            );
        }

        // Destroyed: every angle claim. Unlike a length, an angle has no
        // direction it can be accidentally parallel to, so a shear takes all of
        // them -- the right angle, its cosine, and the perpendicularity of the
        // same two segments. A shear turns a right angle into an oblique one
        // and the dot product stops being zero.
        let right_angle = Constraint::RightAngle {
            at: Angle3::new("A", "B", "C"),
        };
        assert!(!report.graph.has_fact(&right_angle));
        assert!(report
            .losses_of(PredicateClass::Angle)
            .iter()
            .any(|loss| loss.reason == DropReason::Refuted));
        // The perpendicularity of the same two segments goes with it.
        assert!(!report.graph.has_fact(&Constraint::Perpendicular {
            first: seg("A", "B"),
            second: seg("A", "C")
        }));
        // The exact cosine of 0 is no longer the cosine of that angle.
        assert!(!report.graph.has_fact(&Constraint::AngleIs {
            at: Angle3::new("A", "B", "C"),
            cos: QSqrt::rational(f(0, 1))
        }));
        // The area comparison survives, because this shear has determinant 1.
        assert!(report.graph.has_fact(&Constraint::AreaEqual {
            first: tri("A", "B", "C"),
            second: tri("A", "B", "F")
        }));
        // The two stated lengths, and the difference between them is the whole
        // character of a shear. `A(0,0) B(4,0)` runs *along* the shear's axis
        // and is genuinely left alone, so its claim survives and pretending
        // otherwise would be the over-claim this module exists to stop.
        assert!(
            report.graph.has_fact(&Constraint::LengthIs {
                seg: seg("A", "B"),
                square: f(16, 1)
            }),
            "a segment along the shear's axis really does keep its length"
        );
        // `A(0,0) C(0,4)` runs *across* it, is stretched, and its stated
        // length is refuted by the exact predicate.
        assert!(
            !report.graph.has_fact(&Constraint::LengthIs {
                seg: seg("A", "C"),
                square: f(16, 1)
            }),
            "a segment across the shear's axis must lose its stated length"
        );
        assert!(report.dropped.iter().any(|loss| {
            loss.class == PredicateClass::Length && loss.reason == DropReason::Refuted
        }));
        // And no payload was rewritten anywhere in this example, because a
        // shear has no uniform factor to rewrite one with. Restating `AC` by
        // `1 + k^2` would be right by accident for this segment and wrong for
        // `AB`, which is the worst kind of right.
        assert!(report.retargeted.is_empty(), "{:?}", report.retargeted);
        // `Triangle` needs a non-degenerate triangle, and a shear of a
        // non-degenerate triangle is non-degenerate: incidence is preserved.
        assert!(report.graph.has_fact(&Constraint::Triangle {
            a: "A".into(),
            b: "B".into(),
            c: "C".into()
        }));
    }

    #[test]
    fn test_shear_breaks_a_stated_squared_length_exactly() {
        // The length claim has to go, and the reason it goes must be a
        // refutation by the exact predicate rather than a bookkeeping entry.
        let graph = graph_from_points(
            vec![p("A", 0, 0), p("B", 0, 4), p("C", 1, 0)],
            vec![
                Constraint::LengthIs {
                    seg: seg("A", "B"),
                    square: f(16, 1),
                },
                Constraint::Collinear {
                    a: "A".into(),
                    b: "C".into(),
                    c: "C".into(),
                },
            ],
        );
        // A vertical segment of squared length 16: shearing in x by k = 1 sends
        // B to (4, 4), so the squared length is 32 and 16 is now false.
        let report = Transform::shear(Q::ONE).apply_reported(&graph).unwrap();
        let length_claim = Constraint::LengthIs {
            seg: seg("A", "B"),
            square: f(16, 1),
        };
        let loss = report
            .losses_of(PredicateClass::Length)
            .into_iter()
            .find(|loss| loss.constraint == length_claim)
            .expect("the length claim to be dropped");
        assert_eq!(loss.reason, DropReason::Refuted);
        // The claim was carried *verbatim*, not restated by some global factor:
        // a shear has no single length scale, and inventing one is how a
        // sheared figure ends up carrying a length claim that quietly refers to
        // a different measurement.
        assert_eq!(loss.constraint, length_claim);
        // The original figure really did have that length, exactly.
        assert!(graph.holds(&length_claim).unwrap());
        // And the transformed coordinates confirm it by hand: the new squared
        // length is exactly 32, not 16 and not "about 16".
        let (bx, by) = report.graph.coords("B").unwrap();
        assert_eq!((bx, by), (q(4, 1), q(4, 1)));
        assert_eq!(
            bx.mul(&bx).unwrap().add(&by.mul(&by).unwrap()).unwrap(),
            q(32, 1)
        );
    }

    #[test]
    fn test_a_sheared_circle_is_refused_as_an_ellipse_rather_than_carried() {
        // A circle is a centre and a squared radius. A shear's image is an
        // ellipse, which this kernel has no word for, so the honest outcome is
        // a refusal with the reason attached -- not a circle with a radius the
        // figure no longer has.
        let mut graph = triangle_scene();
        graph
            .add_circle(KCircle {
                name: "w".into(),
                center: "A".into(),
                radius_sq: f(16, 1),
            })
            .unwrap();
        // The claims that depend on the object, not just the object itself: a
        // circle nothing states anything about is not a fact to lose.
        assert!(graph.add_fact(
            Constraint::Circle {
                name: "w".into(),
                center: "A".into(),
                radius_sq: f(16, 1)
            },
            Provenance::Given
        ));
        assert!(graph.add_fact(
            Constraint::OnCircle {
                p: "B".into(),
                circle: "w".into()
            },
            Provenance::Given
        ));
        // `A(0,0)` with radius 4 needs points on *opposite* ends, so the
        // diameter runs to a point the fixture does not have yet.
        graph.points.push(KPoint {
            name: "G".into(),
            x: f(-4, 1),
            y: f(0, 1),
        });
        assert!(graph.add_fact(
            Constraint::Diameter {
                circle: "w".into(),
                a: "B".into(),
                b: "G".into()
            },
            Provenance::Given
        ));
        let report = Transform::shear(Q::ONE).apply_reported(&graph).unwrap();
        // All three circle claims go, and none of them is a *point* incidence
        // that happened to be re-decided: the ellipse refusal is a separate
        // reason, recorded as one.
        assert_eq!(
            report
                .dropped
                .iter()
                .filter(|loss| loss.reason == DropReason::CircleBecomesEllipse)
                .count(),
            3
        );
        assert!(
            report.graph.circles.is_empty(),
            "no circle may be carried through a shear"
        );
        let loss = report
            .dropped
            .iter()
            .find(|loss| loss.reason == DropReason::CircleBecomesEllipse)
            .expect("the circle claim to be dropped as an ellipse");
        assert_eq!(loss.class, PredicateClass::Length);
        // The kernel agrees the diameter was real before the shear: `B(4,0)`
        // and `G(-4,0)` are both 4 from `A(0,0)` and their midpoint is `A`.
        assert!(graph
            .holds(&Constraint::Diameter {
                circle: "w".into(),
                a: "B".into(),
                b: "G".into()
            })
            .unwrap());
        assert!(report
            .graph
            .not_established
            .iter()
            .any(|note| note.contains("ellipse")));
        // Under a *similarity* the same circle is carried, with its radius
        // rescaled by the square of the scale factor: 16 * (3/2)^2 = 36.
        let scaled = Transform::scaling(q(3, 2), Q::ONE)
            .unwrap()
            .apply_reported(&graph)
            .unwrap();
        assert_eq!(scaled.graph.circles.len(), 1);
        assert_eq!(scaled.graph.circles[0].radius_sq, f(36, 1));
        assert_eq!(scaled.graph.circles[0].center, "A");
    }

    // ---------------------------------------------------------- permutation --

    #[test]
    fn test_permuting_vertex_names_yields_an_isomorphic_graph() {
        // `A -> B -> C -> A` on a three-point figure: same coordinates, same
        // coordinates under new names, same fact count. This is the family that
        // demonstrates the proof graph and not the drawing is the object -- and
        // it is also the family the module doc calls duplication, which is why
        // it is here as the identity the others are measured against.
        let graph = graph_from_points(
            vec![p("A", 0, 0), p("B", 4, 0), p("C", 0, 2)],
            vec![
                Constraint::Collinear {
                    a: "A".into(),
                    b: "B".into(),
                    c: "B".into(),
                },
                Constraint::Distinct {
                    a: "A".into(),
                    b: "C".into(),
                },
                Constraint::LengthIs {
                    seg: seg("A", "B"),
                    square: f(16, 1),
                },
            ],
        );
        let rename: BTreeMap<String, String> = [("A", "B"), ("B", "C"), ("C", "A")]
            .iter()
            .map(|(from, to)| (from.to_string(), to.to_string()))
            .collect();
        let report = Transform::permutation(rename)
            .apply_reported(&graph)
            .unwrap();
        assert!(report.kept_everything(), "{}", report.summary());
        // Same number of facts, and every one of them still holds of the
        // renamed figure.
        assert_eq!(report.graph.facts.len(), graph.facts.len());
        for fact in &report.graph.facts {
            assert!(
                constraint_holds_in(&[], &coord_table(&report.graph), &fact.constraint).unwrap(),
                "{} should hold of the permuted figure",
                fact.constraint.describe()
            );
        }
        // The coordinates did not move -- the names did. `A` now sits where `C`
        // was, and `B` where `A` was, which is the whole content of a
        // relabelling: the same figure, none of whose names stayed put.
        assert_eq!(
            report.graph.coords("A").unwrap(),
            graph.coords("C").unwrap()
        );
        assert_eq!(
            report.graph.coords("B").unwrap(),
            graph.coords("A").unwrap()
        );
        assert_eq!(
            report.graph.coords("C").unwrap(),
            graph.coords("B").unwrap()
        );
    }

    #[test]
    fn test_a_permutation_that_is_not_a_bijection_is_refused() {
        let graph = graph_from_points(vec![p("A", 0, 0), p("B", 1, 0), p("C", 0, 1)], vec![]);
        // Two sources reaching one target: `A` and `B` both rename to `B`, so
        // the figure would come to have two points called B.
        let squashed: BTreeMap<String, String> = [("A", "B"), ("B", "B"), ("C", "C")]
            .iter()
            .map(|(x, y)| (x.to_string(), y.to_string()))
            .collect();
        let refused = refusal(Transform::permutation(squashed).apply(&graph));
        assert!(
            matches!(refused, GeometryError::RepeatedPoint { .. }),
            "{refused}"
        );
        // A three-cycle is *not* refused: `A->B, B->C, C->A` moves every name
        // and keeps the figure intact, and a check that rejected it would be
        // rejecting the permutation family outright.
        let cycle: BTreeMap<String, String> = [("A", "B"), ("B", "C"), ("C", "A")]
            .iter()
            .map(|(x, y)| (x.to_string(), y.to_string()))
            .collect();
        assert!(Transform::permutation(cycle).apply(&graph).is_ok());
        // A *source* the scene does not declare is refused, and names it.
        let unknown: BTreeMap<String, String> = [("Q", "A")]
            .iter()
            .map(|(x, y)| (x.to_string(), y.to_string()))
            .collect();
        let missing = refusal(Transform::permutation(unknown).apply(&graph));
        assert!(
            matches!(missing, GeometryError::EmptyGeometry(ref detail) if detail.contains('Q')),
            "{missing}"
        );
        // Renaming *onto* a fresh name is not a collision and is allowed: `A`
        // moving to `Z` leaves nobody called `Z` behind.
        let fresh: BTreeMap<String, String> = [("A", "Z"), ("B", "C"), ("C", "A")]
            .iter()
            .map(|(x, y)| (x.to_string(), y.to_string()))
            .collect();
        assert!(Transform::permutation(fresh).apply(&graph).is_ok());
        // A rename onto a point that *keeps* its own name is refused too: the
        // figure would come to have two points called B.
        let onto: BTreeMap<String, String> = [("A", "B")]
            .iter()
            .map(|(x, y)| (x.to_string(), y.to_string()))
            .collect();
        let clobbered = refusal(Transform::permutation(onto).apply(&graph));
        assert!(
            matches!(clobbered, GeometryError::RepeatedPoint { .. }),
            "{clobbered}"
        );
    }

    // ------------------------------------------------------------ the report --

    #[test]
    fn test_the_confusion_matrix_adds_up() {
        // `kept + lost == examined * trials` is the identity that makes the
        // matrix checkable rather than decorative. A tally that quietly loses a
        // fact would still *look* like a report.
        let graph = triangle_scene();
        let report = invariance_report(&graph, 3).unwrap();
        assert!(report.is_consistent());
        for cell in &report.cells {
            assert!(
                cell.is_consistent(),
                "{} / {} does not add up",
                cell.transform,
                cell.class
            );
        }
        // The families that ran: every one of them, at least once.
        for family in TransformFamily::ALL {
            let trials = report.row(family).first().map_or(0, |cell| cell.trials);
            assert!(trials > 0, "{family} ran no trials at all");
        }
    }

    #[test]
    fn test_the_exact_families_preserve_everything_and_the_shear_does_not() {
        // The verdict, read as a test. Every similarity family keeps every
        // class; the shear destroys the metric classes and touches nothing
        // else. This is the claim the module doc makes, checked.
        let graph = triangle_scene();
        let report = invariance_report(&graph, 3).unwrap();
        for family in TransformFamily::ALL.iter().filter(|f| f.is_exact()) {
            for cell in report.row(*family) {
                if cell.is_vacuous() {
                    continue;
                }
                assert!(
                    cell.preserved_all,
                    "{family} lost a {} fact: {}/{} kept",
                    cell.class,
                    cell.kept,
                    cell.kept + cell.lost
                );
            }
        }
        // The shear: metric classes gone, structural classes intact.
        for cell in report.row(TransformFamily::Shear) {
            if cell.is_vacuous() {
                continue;
            }
            if InvarianceReport::shear_destroys(cell.class) {
                // `refuted_any`, not `refuted_all`: a metric claim whose
                // segments run along the shear's axis survives by accident, and
                // the family is not to be condemned for the fixture's accident.
                assert!(
                    cell.refuted_any,
                    "the shear broke nothing at all among the {} facts",
                    cell.class
                );
            } else {
                assert!(
                    cell.preserved_all,
                    "the shear broke a {} fact, which it must not",
                    cell.class
                );
            }
        }
        // And the verdict says so in words, with no "NOT as claimed" in it.
        let verdict = report.verdict();
        assert!(!verdict.contains("NOT as claimed"), "{verdict}");
        assert!(verdict.contains("as claimed"), "{verdict}");
    }

    #[test]
    fn test_the_shear_actually_teaches_the_lesson_it_claims_to() {
        // A scene whose only metric fact is a length: the shear must catch a
        // model that answers the sheared figure from a memorized template. The
        // discrimination sentence is the report's way of saying that, and it
        // must be present here and absent from a scene with nothing to refute.
        let graph = graph_from_points(
            vec![p("A", 0, 0), p("B", 0, 4)],
            vec![Constraint::LengthIs {
                seg: seg("A", "B"),
                square: f(16, 1),
            }],
        );
        let report = invariance_report(&graph, 2).unwrap();
        let verdict = report.verdict();
        assert!(
            verdict.contains("a model answering a sheared figure with the unstretched answer"),
            "{verdict}"
        );

        // The honest counterpart: a scene with no metric fact has nothing for
        // the shear to refute, and the report says so instead of claiming a
        // discrimination it did not demonstrate.
        let plain = graph_from_points(
            vec![p("A", 0, 0), p("B", 2, 0), p("C", 0, 2)],
            vec![Constraint::Collinear {
                a: "A".into(),
                b: "B".into(),
                c: "B".into(),
            }],
        );
        let report = invariance_report(&plain, 2).unwrap();
        assert!(
            !report.verdict().contains("memorized template"),
            "{}",
            report.verdict()
        );
    }

    #[test]
    fn test_confidence_is_carried_and_never_rounded_up() {
        // Audit fix 10: a grade below 1.0 is carried through an augmentation, not
        // promoted because the transform happened to keep the fact.
        let mut graph = triangle_scene();
        // Pushed rather than added: `add_fact` refuses a duplicate predicate,
        // and the figure already states this collinearity as `given`. The
        // graded copy is a second, weaker reading of the same stroke, which is
        // exactly the case audit fix 10 is about.
        graph.facts.push(Fact::observed(
            Constraint::Collinear {
                a: "A".into(),
                b: "D".into(),
                c: "B".into(),
            },
            "observed:0.83",
            0.83,
        ));
        assert!((scene_confidence(&graph) - 0.83).abs() < 1e-12);
        let report = Transform::quarter_turn().apply_reported(&graph).unwrap();
        assert!((report.confidence_before - 0.83).abs() < 1e-12);
        assert!((report.confidence_after - 0.83).abs() < 1e-12);
        // The graded fact really is still graded in the transformed scene.
        let graded = report
            .graph
            .facts
            .iter()
            .filter(|fact| fact.constraint.describe() == "collinear(A,D,B)")
            .map(|fact| fact.confidence)
            .collect::<Vec<_>>();
        assert!(
            graded
                .iter()
                .any(|confidence| (confidence - 0.83).abs() < 1e-12),
            "the observed collinearity lost its grade; confidences were {graded:?}"
        );
        // The certain copy is still certain: an augmentation neither promotes
        // the weak reading nor knocks down the strong one.
        assert!(
            graded.iter().any(|confidence| *confidence >= 1.0),
            "{graded:?}"
        );
        // An empty ledger is certainty, because nothing has been doubted.
        let empty = graph_from_points(vec![p("A", 0, 0)], vec![]);
        assert_eq!(scene_confidence(&empty), 1.0);
    }

    #[test]
    fn test_a_dependency_naming_a_dropped_fact_is_not_left_dangling() {
        // A certificate that names a fact the graph no longer states is worse
        // than one naming fewer facts, so a permutation that drops a premise
        // drops the dependency with it.
        let mut graph = triangle_scene();
        // The premise is a real fact of the figure; the conclusion is a real
        // fact too, derived from it, so the permutation has a live dependency
        // to carry and the assertion is about the carrying. The dependency is
        // named by a fact's *description*, so after a `A->B->C->A` cycle the
        // premise reads `midpoint(D)=mid(B,C)` -- the renamed premise, not the
        // old one. A permutation that failed to rename it would leave a
        // certificate pointing at a fact the graph never stated.
        let premise = Constraint::MidpointOf {
            p: "D".into(),
            a: "A".into(),
            b: "B".into(),
        };
        let conclusion = Constraint::Distinct {
            a: "E".into(),
            b: "C".into(),
        };
        graph.facts.retain(|fact| fact.constraint != conclusion);
        graph.facts.push(Fact::derived(
            conclusion.clone(),
            "given-figure",
            vec![premise.describe()],
        ));
        let rename: BTreeMap<String, String> = [("A", "B"), ("B", "C"), ("C", "A")]
            .iter()
            .map(|(x, y)| (x.to_string(), y.to_string()))
            .collect();
        let report = Transform::permutation(rename)
            .apply_reported(&graph)
            .unwrap();
        // The conclusion is looked for under the name the permutation gave it
        // (`C` became `A`), not the one it started with: a relabelling that
        // left the *statement* unrenamed would be a figure with a fact about a
        // point that no longer exists.
        let carried_conclusion = Constraint::Distinct {
            a: "E".into(),
            b: "A".into(),
        };
        let carried = report
            .graph
            .facts
            .iter()
            .find(|fact| fact.constraint == carried_conclusion)
            .expect("the conclusion to survive under its new name");
        // The dependency was renamed along with the premise, and every name it
        // carries is a fact the transformed graph actually states.
        assert_eq!(carried.depends_on.len(), 1, "the dependency was lost");
        for dep in &carried.depends_on {
            assert!(
                report.graph.facts.iter().any(|fact| fact.key() == *dep),
                "dangling dependency {dep}"
            );
        }
        // And the name it carries is the *renamed* premise, which is the whole
        // point: `midpoint(D)=mid(A,B)` is not a fact of the permuted graph.
        assert_eq!(carried.depends_on[0], "midpoint(D)=mid(B,C)");
    }

    // ------------------------------------------------------------- examples --

    #[test]
    fn test_augmentations_give_distinct_presentations_of_one_graph() {
        let graph = triangle_scene();
        let claim = Constraint::MidpointOf {
            p: "D".into(),
            a: "A".into(),
            b: "B".into(),
        };
        let examples = augmentations(&graph, claim.clone(), 8).unwrap();
        assert!(examples.len() >= 5, "only {} presentations", examples.len());
        // No two examples are the same picture: the generator filters on the
        // exact coordinates and names, so this is a real distinctness claim.
        let mut signatures: Vec<String> = Vec::new();
        for example in &examples {
            let signature = presentation_signature(&example.graph).unwrap();
            assert!(
                !signatures.contains(&signature),
                "a duplicate presentation appeared"
            );
            signatures.push(signature);
        }
        // The claim is carried across and is true of every presentation that
        // kept everything -- a shear presentation of a midpoint still has it,
        // because a midpoint is affine invariant.
        for example in &examples {
            assert!(
                constraint_holds_in(
                    &example.graph.circles,
                    &coord_table(&example.graph),
                    &example.claim
                )
                .unwrap(),
                "{} carries a claim its own figure does not satisfy",
                example.presentation
            );
        }
        // And the families are varied, which is the difference between a set of
        // presentations and a set of translations.
        let families: Vec<TransformFamily> =
            examples.iter().map(|e| e.transform.family()).collect();
        assert!(families.contains(&TransformFamily::Rotation));
        assert!(families.contains(&TransformFamily::Reflection));
        assert!(
            families.len() > 2
                && families
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    >= 3
        );
    }

    #[test]
    fn test_an_augmented_example_says_whether_it_is_still_the_same_problem() {
        // The pairing a trainer consumes. An equivalent presentation (a
        // similarity) has a claim that still holds; a sheared one does not, and
        // must not be handed the original answer.
        let graph = graph_from_points(
            vec![p("A", 0, 0), p("B", 0, 4), p("C", 2, 0)],
            vec![
                Constraint::LengthIs {
                    seg: seg("A", "B"),
                    square: f(16, 1),
                },
                Constraint::Collinear {
                    a: "A".into(),
                    b: "B".into(),
                    c: "B".into(),
                },
            ],
        );
        let claim = Constraint::LengthIs {
            seg: seg("A", "B"),
            square: f(16, 1),
        };
        // The catalogue is walked family by family and the shear is last, so a
        // small `count` stops before reaching it. Asking for the whole thing is
        // the honest way to say "I want the shear included".
        let examples = augmentations(&graph, claim.clone(), 64).unwrap();
        assert!(!examples.is_empty());
        let sheared: Vec<&AugmentedExample> = examples
            .iter()
            .filter(|e| e.transform.family() == TransformFamily::Shear)
            .collect();
        assert!(!sheared.is_empty(), "no sheared example was produced");
        for example in &sheared {
            assert!(!example.claim_holds, "a sheared length claim must not hold");
            assert!(!example.is_equivalent());
            assert!(!example.refuted.is_empty());
            assert_eq!(example.claim_class(), PredicateClass::Length);
        }
        let similar: Vec<&AugmentedExample> = examples
            .iter()
            .filter(|e| e.transform.preserves_every_fact())
            .collect();
        assert!(!similar.is_empty());
        for example in &similar {
            assert!(example.claim_holds, "a similarity must keep the claim true");
            assert!(example.is_equivalent(), "{}", example.presentation);
        }
    }

    #[test]
    fn test_a_transform_survives_a_json_round_trip() {
        // `Frac` holds an `i128`, which `serde_json` refuses without its
        // `arbitrary_precision` feature -- a property of this crate's
        // `serde_json`, not of the transform. The round trip is therefore
        // checked through the wire type directly, which is the same
        // `From`/`TryFrom` pair the serde impls use.
        // A dataset stores these as JSON, so the round trip has to be exact and
        // it has to be *validated*: a file carrying an off-unit-circle rotation
        // is refused at the boundary rather than applied to a figure.
        let transforms = vec![
            Transform::quarter_turn(),
            Transform::three_four_five_turn(),
            Transform::translation(q(3, 7), q(-1, 2)),
            Transform::scaling(q(3, 2), q(1, 5)).unwrap(),
            Transform::reflection_over_x_axis(),
            Transform::reflection_over_points("A", "B"),
            Transform::shear(q(-3, 4)),
        ];
        for transform in transforms {
            let wire = TransformWire::from(&transform);
            let back = Transform::try_from(wire).unwrap();
            assert_eq!(back, transform, "round trip changed {transform}");
        }
        // A hand-written transform off the unit circle is refused on the way
        // in, at the same place the deserializer would refuse it.
        let bad = TransformWire::Rotation {
            cos: f(3, 5),
            sin: f(3, 5),
        };
        assert!(matches!(
            refusal(Transform::try_from(bad)),
            GeometryError::NoRationalSolution(_)
        ));
        // And a negative scale is refused there too.
        let negative = TransformWire::Scaling {
            num: f(-1, 1),
            den: f(2, 1),
        };
        assert!(matches!(
            refusal(Transform::try_from(negative)),
            GeometryError::NoRationalSolution(_)
        ));
    }

    // ------------------------------------------------------------- utilities --

    /// A coordinate table for a scene, which is what the kernel's predicates
    /// consume.
    fn coord_table(graph: &SceneGraph) -> Vec<(String, Q, Q)> {
        graph
            .points
            .iter()
            .map(|point| {
                (
                    point.name.clone(),
                    point.x.to_q().unwrap(),
                    point.y.to_q().unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn test_the_affine_map_reports_its_own_length_scale_exactly() {
        // The number a squared length gets multiplied by, checked per family.
        let graph = triangle_scene();
        let cases: [(Transform, Q); 5] = [
            (Transform::quarter_turn(), q(1, 1)),
            (Transform::half_turn(), q(1, 1)),
            (Transform::translation(q(5, 1), q(5, 1)), q(1, 1)),
            (Transform::reflection_over_x_axis(), q(1, 1)),
            (Transform::scaling(q(3, 2), Q::ONE).unwrap(), q(9, 4)),
        ];
        for (transform, want) in cases {
            let affine = transform.affine(&graph).unwrap();
            assert_eq!(affine.squared_length_scale().unwrap(), want, "{transform}");
            assert!(
                affine.is_similarity().unwrap(),
                "{transform} should be a similarity"
            );
        }
        // The shear is the one that is not, and its scale is not a length scale
        // at all: `1 + k^2` is what a *horizontal* segment picks up, which is
        // precisely why it must not be used to rewrite a stated length.
        let shear = Transform::shear(Q::ONE).affine(&graph).unwrap();
        assert!(!shear.is_similarity().unwrap());
        assert_eq!(shear.determinant().unwrap(), q(1, 1));
        // Every other family has determinant of magnitude 1, as a similarity must.
        for transform in [
            Transform::quarter_turn(),
            Transform::half_turn(),
            Transform::reflection_over_x_axis(),
        ] {
            let determinant = transform.affine(&graph).unwrap().determinant().unwrap();
            assert!(
                determinant == q(1, 1) || determinant == q(-1, 1),
                "{transform} has determinant {determinant}"
            );
        }
    }
}
