//! The deterministic geometry kernel (roadmap Phase 33): the source of
//! mathematical truth the neural reasoner sits on top of.
//!
//! The division of labor is the one the field converged on with
//! AlphaGeometry-class systems: the neural model proposes; the kernel proves,
//! refutes, or refuses. Nothing in this module uses floating point where a
//! fact is at stake -- quantities are exact rationals ([`Q`], i128
//! numerator and denominator), so "collinear" means *collinear*, not
//! "collinear to within epsilon".
//!
//! # What lives here
//!
//! - [`Q`]: exact rational arithmetic (normalized fractions, exact compare).
//! - [`SceneGraph`]: the canonical typed IR -- points, constraints, and the
//!   provenance of every fact (`given` / `derived{rule}` / `constructed{op}`),
//!   plus the assumption ledger (`established` vs `not_established`).
//!   Serde JSON, no Burn dependency: this is the file format an engine routes
//!   from.
//! - Non-degeneracy at every operation: parallel or coincident lines refuse
//!   to intersect, degenerate segments refuse to define directions -- the
//!   kernel returns errors instead of silently propagating invalid
//!   configurations.
//! - [`Rule`]: deduction rules with typed preconditions; applying one emits a
//!   fact whose provenance names the rule and its inputs, so every proof step
//!   carries a machine-checkable certificate.
//! - [`construct`]: exact constructions (midpoint, line intersection,
//!   projection, reflection) over the typed operation grammar.
//! - [`falsify`]: the counterexample generator -- sample random configurations
//!   satisfying the premises exactly; one violating the claim rejects it.
//!   Many satisfying draws are evidence, never proof.
//!
//! Scope, stated plainly, because the audit was about claims outrunning
//! machinery. This file is the kernel: exact objects, exact predicates, a rule
//! library that carries its own side conditions, construction with exact
//! coordinates, saturation that logs a dependency-bearing proof trace, and a
//! premise-guided falsifier. Around it sit the pieces the audit named, each in
//! its own module so the boundary stays honest: `geomproof` for construction
//! beam search, machine-checkable certificates, the independent verifier and
//! the Lean 4 export; `geomnegatives` for the false derivations a proof head
//! has to learn to reject; `geomroute` for the MoSME-style
//! expert-per-failure-mode controller that gates every proposal behind a
//! verifier; `geomexercises` for the difficulty-ordered exercises that feed
//! them; `geom3d` and `geomnonclid` for the incidence models outside the plane
//! and outside Euclid; `geomvision` for diagram-to-graph grounding;
//! `geomaugment` for structure-preserving transforms. What is still *not*
//! here: a Gröbner-complete prover for construction *separation* (the falsifier
//! solves most premises exactly and rejection-samples the rest), hyperbolic
//! trigonometry, and a checked compilation of the exported Lean file. Those are
//! declared where they bite — the sin the audit found was never the ambition,
//! only the prose that ran ahead of it.

use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::fmt;

/// An exact rational: `num / den`, `den > 0`, always normalized (gcd 1).
/// The kernel's only number type where facts are at stake.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Q {
    pub num: i128,
    pub den: i128,
}

fn gcd_i128(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

impl Q {
    pub const ZERO: Q = Q { num: 0, den: 1 };
    pub const ONE: Q = Q { num: 1, den: 1 };

    pub fn new(num: i128, den: i128) -> anyhow::Result<Self> {
        anyhow::ensure!(den != 0, "a rational denominator is zero");
        let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
        let g = gcd_i128(num, den).max(1);
        Ok(Self {
            num: num / g,
            den: den / g,
        })
    }

    pub fn from_int(value: i64) -> Self {
        Self {
            num: i128::from(value),
            den: 1,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.num == 0
    }

    pub fn add(&self, other: &Self) -> anyhow::Result<Self> {
        let left = self
            .num
            .checked_mul(other.den)
            .ok_or_else(|| anyhow::anyhow!("rational overflow in addition"))?;
        let right = other
            .num
            .checked_mul(self.den)
            .ok_or_else(|| anyhow::anyhow!("rational overflow in addition"))?;
        let sum = left
            .checked_add(right)
            .ok_or_else(|| anyhow::anyhow!("rational overflow in addition"))?;
        let den = self
            .den
            .checked_mul(other.den)
            .ok_or_else(|| anyhow::anyhow!("rational overflow in addition"))?;
        Self::new(sum, den)
    }

    pub fn sub(&self, other: &Self) -> anyhow::Result<Self> {
        self.add(&other.neg()?)
    }

    pub fn neg(&self) -> anyhow::Result<Self> {
        Self::new(-self.num, self.den)
    }

    pub fn mul(&self, other: &Self) -> anyhow::Result<Self> {
        let num = self
            .num
            .checked_mul(other.num)
            .ok_or_else(|| anyhow::anyhow!("rational overflow in multiplication"))?;
        let den = self
            .den
            .checked_mul(other.den)
            .ok_or_else(|| anyhow::anyhow!("rational overflow in multiplication"))?;
        Self::new(num, den)
    }

    pub fn div(&self, other: &Self) -> anyhow::Result<Self> {
        anyhow::ensure!(!other.is_zero(), "division by the zero rational");
        let num = self
            .num
            .checked_mul(other.den)
            .ok_or_else(|| anyhow::anyhow!("rational overflow in division"))?;
        let den = self
            .den
            .checked_mul(other.num)
            .ok_or_else(|| anyhow::anyhow!("rational overflow in division"))?;
        Self::new(num, den)
    }

    pub fn half(&self) -> anyhow::Result<Self> {
        self.div(&Q::from_int(2))
    }

    pub fn to_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    pub fn less(&self, other: &Self) -> bool {
        // both denominators are positive, so the cross product orders exactly
        self.num * other.den < other.num * self.den
    }
}

impl fmt::Display for Q {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

// ------------------------------------------------------------------ QSqrt --

/// An exact element of a quadratic field: `a + b * sqrt(d)`, with `d` a
/// square-free non-negative integer and `d == 0` meaning the rational `a`.
///
/// Audit fix 7 is the reason this type exists. `~ 60` where the truth is
/// `= 60` is the soundest kind of bug a geometry system can have, and it comes
/// from `arccos` being the only language the old kernel had for angles. Cosines
/// need no such escape: with legs `u` and `v`,
/// `cos = dot(u,v) / sqrt(|u|^2 |v|^2)`, and pulling the square part out of the
/// rational `|u|^2 |v|^2` leaves exactly this shape for every lattice angle.
/// So `cos(60 deg) = 1/2` and `cos(45 deg) = 1/2*sqrt(2)`, stored, compared and
/// serialized without one floating-point operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QSqrt {
    pub a: Frac,
    pub b: Frac,
    /// The square-free radicand; `0` marks the rational case `b == 0`.
    pub d: i64,
}

/// `value == square * square_free`, both factors positive and the second
/// square-free. Trial division: exact, and refused above one trillion rather
/// than quietly approximated. Every radicand the kernel produces comes from a
/// lattice point or a small exact solver, so hitting that limit means something
/// upstream has already gone wrong.
fn squarefree_split(value: i64) -> anyhow::Result<(i64, i64)> {
    anyhow::ensure!(
        value >= 1,
        "a square-free split needs a positive integer, got {value}"
    );
    let limit = 1_000_000i64;
    anyhow::ensure!(
        value <= limit * limit,
        "refusing to factor {value} exactly: above {limit}^2 the kernel would have to guess"
    );
    let (mut rest, mut square, mut square_free) = (value, 1i64, 1i64);
    let mut p = 2i64;
    while p * p <= rest {
        let mut count = 0i32;
        while rest % p == 0 {
            rest /= p;
            count += 1;
        }
        if count % 2 == 1 {
            square_free = square_free
                .checked_mul(p)
                .context(GeometryError::ArithmeticOverflow)?;
        }
        for _ in 0..count / 2 {
            square = square
                .checked_mul(p)
                .context(GeometryError::ArithmeticOverflow)?;
        }
        p += 1;
    }
    if rest > 1 {
        square_free = square_free
            .checked_mul(rest)
            .context(GeometryError::ArithmeticOverflow)?;
    }
    Ok((square, square_free))
}

/// `-1`, `0`, `1` for a rational.
fn sign_of(q: &Q) -> i8 {
    if q.is_zero() {
        0
    } else if q.less(&Q::ZERO) {
        -1
    } else {
        1
    }
}

impl QSqrt {
    /// The rational `a`, with the radical part dropped.
    pub fn rational(a: Frac) -> Self {
        Self {
            a,
            b: Frac::from_int(0),
            d: 0,
        }
    }

    /// `a + b sqrt d`, normalized: a zero radical or a zero radicand collapses
    /// to the rational case, and square factors inside the radicand fold into
    /// `b`. Equal values always produce equal fields, which is what makes `==`
    /// on this type a decision rather than a comparison of approximations.
    pub fn new(a: Frac, b: Frac, d: i64) -> anyhow::Result<Self> {
        anyhow::ensure!(d >= 0, "a radicand cannot be negative (got {d})");
        let (a_q, b_q) = (a.to_q()?, b.to_q()?);
        if b_q.is_zero() || d == 0 {
            return Ok(Self::rational(a));
        }
        let (square, square_free) = squarefree_split(d)?;
        let folded = if square == 1 {
            b_q
        } else {
            b_q.mul(&Q::from_int(square))?
        };
        if square_free == 1 {
            return Ok(Self::rational(Frac::from_q(a_q.add(&folded)?)));
        }
        Ok(Self {
            a,
            b: Frac::from_q(folded),
            d: square_free,
        })
    }

    /// The pure radical `sqrt(d)`.
    pub fn radical(d: i64) -> anyhow::Result<Self> {
        Self::new(Frac::from_int(0), Frac::from_int(1), d)
    }

    pub fn is_rational(&self) -> bool {
        self.d == 0
    }

    /// The value as a rational, when there is no radical part.
    pub fn rational_value(&self) -> anyhow::Result<Option<Q>> {
        Ok(if self.d == 0 {
            Some(self.a.to_q()?)
        } else {
            None
        })
    }
}

impl QSqrt {
    pub fn add(&self, other: &Self) -> anyhow::Result<Self> {
        anyhow::ensure!(
            self.d == other.d,
            "adding {self} and {other} exactly needs a biquadratic field, which this kernel does not have"
        );
        Self::new(
            Frac::from_q(self.a.to_q()?.add(&other.a.to_q()?)?),
            Frac::from_q(self.b.to_q()?.add(&other.b.to_q()?)?),
            self.d,
        )
    }

    pub fn neg(&self) -> anyhow::Result<Self> {
        Self::new(
            Frac::from_q(self.a.to_q()?.neg()?),
            Frac::from_q(self.b.to_q()?.neg()?),
            self.d,
        )
    }

    pub fn sub(&self, other: &Self) -> anyhow::Result<Self> {
        self.add(&other.neg()?)
    }

    /// `k * (a + b sqrt d)` for any rational `k`.
    pub fn scale(&self, k: &Q) -> anyhow::Result<Self> {
        Self::new(
            Frac::from_q(self.a.to_q()?.mul(k)?),
            Frac::from_q(self.b.to_q()?.mul(k)?),
            self.d,
        )
    }

    /// `(a + b sqrt d)(c + e sqrt d) = (ac + b e d) + (ae + bc) sqrt d`.
    pub fn mul(&self, other: &Self) -> anyhow::Result<Self> {
        anyhow::ensure!(
            self.d == other.d,
            "multiplying {self} and {other} exactly needs a biquadratic field, which this kernel does not have"
        );
        let (a, b) = (self.a.to_q()?, self.b.to_q()?);
        let (c, e) = (other.a.to_q()?, other.b.to_q()?);
        let d = Q::from_int(self.d);
        let real = a.mul(&c)?.add(&b.mul(&e)?.mul(&d)?)?;
        let radical = a.mul(&e)?.add(&b.mul(&c)?)?;
        Self::new(Frac::from_q(real), Frac::from_q(radical), self.d)
    }

    /// Exact zero test: `a + b sqrt(d)` vanishes only when both parts do, a
    /// square-free radicand making `sqrt(d)` irrational whenever `d >= 2`.
    pub fn is_zero(&self) -> anyhow::Result<bool> {
        Ok(self.a.to_q()?.is_zero() && self.b.to_q()?.is_zero())
    }

    /// The exact sign, `-1`, `0` or `1`. A radical is compared against a
    /// rational by squaring *with the signs tracked*, which loses nothing: two
    /// terms of the same sign cannot cancel, and when they disagree the larger
    /// magnitude wins, decided by `a^2` against `b^2 d` -- still rationals.
    pub fn sign(&self) -> anyhow::Result<i8> {
        let (a, b) = (self.a.to_q()?, self.b.to_q()?);
        if self.d == 0 || b.is_zero() {
            return Ok(sign_of(&a));
        }
        if a.is_zero() {
            return Ok(sign_of(&b));
        }
        let (a_neg, b_neg) = (a.less(&Q::ZERO), b.less(&Q::ZERO));
        if a_neg == b_neg {
            return Ok(if a_neg { -1 } else { 1 });
        }
        let left = a.mul(&a)?;
        let right = b.mul(&b)?.mul(&Q::from_int(self.d))?;
        if left == right {
            return Ok(0);
        }
        Ok(if right.less(&left) {
            if a_neg {
                -1
            } else {
                1
            }
        } else if b_neg {
            -1
        } else {
            1
        })
    }

    /// Exact ordering, defined through the sign of the difference.
    pub fn less(&self, other: &Self) -> anyhow::Result<bool> {
        Ok(self.sub(other)?.sign()? == -1)
    }

    /// Numeric *evidence* for a report or a caption. Never a fact.
    pub fn to_f64(&self) -> anyhow::Result<f64> {
        Ok(self.a.to_q()?.to_f64() + self.b.to_q()?.to_f64() * (self.d as f64).sqrt())
    }
}

impl fmt::Display for Frac {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

impl fmt::Display for QSqrt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display cannot fail, and a Frac is two integers, so the branches here
        // are exhaustive without asking a Result for permission.
        match (self.d == 0, self.a.num == 0) {
            (true, _) => write!(f, "{}", self.a),
            (false, true) => write!(f, "{}*sqrt({})", self.b, self.d),
            (false, false) => write!(f, "{} + {}*sqrt({})", self.a, self.b, self.d),
        }
    }
}

/// A point with exact rational coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KPoint {
    pub name: String,
    pub x: Frac,
    pub y: Frac,
}

/// A JSON-safe rational (the [`Q`] pair, serialized as a two-field object).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Frac {
    pub num: i128,
    pub den: i128,
}

impl Frac {
    pub fn from_q(q: Q) -> Self {
        Self {
            num: q.num,
            den: q.den,
        }
    }

    pub fn to_q(self) -> anyhow::Result<Q> {
        Q::new(self.num, self.den)
    }

    pub fn from_int(value: i64) -> Self {
        Self {
            num: i128::from(value),
            den: 1,
        }
    }
}

impl TryFrom<Frac> for Q {
    type Error = anyhow::Error;

    fn try_from(f: Frac) -> Result<Self, Self::Error> {
        f.to_q()
    }
}

impl From<Q> for Frac {
    fn from(q: Q) -> Self {
        Frac::from_q(q)
    }
}

/// A directed segment between two named points (the kernel's line proxy:
/// a line is a pair of distinct points).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Segment {
    pub from: String,
    pub to: String,
}

/// An angle by name: the vertex, and the two points its legs run to. Naming the
/// vertex explicitly -- rather than writing "angle ABC" and hoping the reader
/// guesses which of the four angles at a crossing is meant -- is what lets a
/// rule, a diagram and a theorem statement agree about one object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Angle3 {
    pub at: String,
    pub from: String,
    pub to: String,
}

impl Angle3 {
    pub fn new(at: &str, from: &str, to: &str) -> Self {
        Self {
            at: at.to_string(),
            from: from.to_string(),
            to: to.to_string(),
        }
    }

    pub fn describe(&self) -> String {
        format!("{}{}{}", self.from, self.at, self.to)
    }
}

/// A triangle named by its three vertices, used by the area and congruence
/// predicates. The order is part of the name: `ABC` and `ACB` are opposite
/// orientations, and a rule that does not care says so out loud.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tri3 {
    pub a: String,
    pub b: String,
    pub c: String,
}

impl Tri3 {
    pub fn new(a: &str, b: &str, c: &str) -> Self {
        Self {
            a: a.to_string(),
            b: b.to_string(),
            c: c.to_string(),
        }
    }

    pub fn describe(&self) -> String {
        format!("{}{}{}", self.a, self.b, self.c)
    }

    pub fn vertices(&self) -> [String; 3] {
        [self.a.clone(), self.b.clone(), self.c.clone()]
    }
}

/// A circle as an object with an identity: a centre named by point and an exact
/// *squared* radius. Squared because the radius of a lattice circle is usually
/// irrational while its square never is, and because a squared radius is what
/// every membership test actually consumes. Audit fix 1: an object with no
/// identity of its own cannot carry provenance, dependencies or confidence,
/// which is precisely how the audit found circles being used as if they did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KCircle {
    pub name: String,
    pub center: String,
    pub radius_sq: Frac,
}

impl KCircle {
    pub fn describe(&self) -> String {
        format!(
            "circle({}:{},r^2={})",
            self.name, self.center, self.radius_sq
        )
    }
}

/// The typed predicates the kernel reasons over.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Constraint {
    Collinear {
        a: String,
        b: String,
        c: String,
    },
    Parallel {
        first: Segment,
        second: Segment,
    },
    Perpendicular {
        first: Segment,
        second: Segment,
    },
    EqualLength {
        first: Segment,
        second: Segment,
    },
    /// `p` is the midpoint of `a` and `b`.
    MidpointOf {
        p: String,
        a: String,
        b: String,
    },
    /// Audit fix 4. Two names, two locations: the assumption that the phrase
    /// "the triangle ABC" smuggles in whenever nobody writes it down.
    Distinct {
        a: String,
        b: String,
    },
    /// Audit fix 4. The three points are not collinear; `ABC` bounds a region.
    NonCollinear {
        a: String,
        b: String,
        c: String,
    },
    /// Audit fix 4. A named triangle that is genuinely a triangle -- the
    /// precondition every triangle theorem in the library now asks for by name.
    Triangle {
        a: String,
        b: String,
        c: String,
    },
    /// `m` lies strictly between `a` and `b`. Order, not only incidence: the
    /// relation betweenness axioms and any honest "the foot falls inside the
    /// segment" both need.
    Between {
        a: String,
        m: String,
        b: String,
    },
    /// `p` divides `ab` in the ratio `num : den`, both positive. A midpoint is
    /// the special case `1 : 1`, and a trisection point is not a mystery.
    RatioOf {
        p: String,
        a: String,
        b: String,
        num: i64,
        den: i64,
    },
    /// The squared length of a segment is a stated rational. Squared, so the
    /// claim stays rational even when the length is not.
    LengthIs {
        seg: Segment,
        square: Frac,
    },
    /// `|second| == (num/den) * |first|`: the scaled form of `EqualLength`, and
    /// the shape the midpoint theorem's second half actually needs.
    ScaleLength {
        first: Segment,
        second: Segment,
        num: i64,
        den: i64,
    },
    /// Two angles are equal, decided on their exact cosines.
    AngleEqual {
        first: Angle3,
        second: Angle3,
    },
    /// A right angle at a named vertex: a zero dot product, so a decision.
    RightAngle {
        at: Angle3,
    },
    /// The exact cosine of a named angle, in `Q(sqrt d)`. This is audit fix 7
    /// turned into a predicate: "the angle at `A` is 60 degrees" is stated as
    /// its cosine `1/2` and checked without leaving the rationals, so it can
    /// never come back as `~ 60`.
    AngleIs {
        at: Angle3,
        cos: QSqrt,
    },
    /// Two triangles of equal area, exact through the doubled shoelace.
    AreaEqual {
        first: Tri3,
        second: Tri3,
    },
    /// Congruence of two triangles, read off their three side lengths.
    Congruent {
        first: Tri3,
        second: Tri3,
    },
    /// Audit fix 1: a circle object, with an identity, a centre and an exact
    /// squared radius -- not three dots that look roughly equidistant.
    Circle {
        name: String,
        center: String,
        radius_sq: Frac,
    },
    /// `p` lies on the named circle.
    OnCircle {
        p: String,
        circle: String,
    },
    /// `a` and `b` are the opposite ends of a diameter of the named circle --
    /// the hypothesis Thales' theorem states, now statable.
    Diameter {
        circle: String,
        a: String,
        b: String,
    },
}

impl Constraint {
    pub fn describe(&self) -> String {
        match self {
            Self::Collinear { a, b, c } => format!("collinear({a},{b},{c})"),
            Self::Parallel { first, second } => {
                format!("parallel({}, {})", first.describe(), second.describe())
            }
            Self::Perpendicular { first, second } => {
                format!("perp({}, {})", first.describe(), second.describe())
            }
            Self::EqualLength { first, second } => {
                format!(
                    "len({}{})=len({}{})",
                    first.from, first.to, second.from, second.to
                )
            }
            Self::MidpointOf { p, a, b } => format!("midpoint({p})=mid({a},{b})"),
            Self::Distinct { a, b } => format!("{a}!={b}"),
            Self::NonCollinear { a, b, c } => format!("noncollinear({a},{b},{c})"),
            Self::Triangle { a, b, c } => format!("triangle({a},{b},{c})"),
            Self::Between { a, m, b } => format!("{m} lies between {a} and {b}"),
            Self::RatioOf { p, a, b, num, den } => {
                format!("{p} divides {a}{b} as {num}:{den}")
            }
            Self::LengthIs { seg, square } => format!("len({})^2={}", seg.describe(), square),
            Self::ScaleLength {
                first,
                second,
                num,
                den,
            } => format!(
                "len({})=({}/{})len({})",
                second.describe(),
                num,
                den,
                first.describe()
            ),
            Self::AngleEqual { first, second } => {
                format!("angle({})=angle({})", first.describe(), second.describe())
            }
            Self::RightAngle { at } => format!("right({})", at.describe()),
            Self::AngleIs { at, cos } => format!("cos(angle({}))={}", at.describe(), cos),
            Self::AreaEqual { first, second } => {
                format!("area({})=area({})", first.describe(), second.describe())
            }
            Self::Congruent { first, second } => {
                format!("congruent({})~({})", first.describe(), second.describe())
            }
            Self::Circle {
                name,
                center,
                radius_sq,
            } => {
                format!("circle({name}) centre {center}, r^2={radius_sq}")
            }
            Self::OnCircle { p, circle } => format!("{p} on circle({circle})"),
            Self::Diameter { circle, a, b } => {
                format!("{a}{b} is a diameter of circle({circle})")
            }
        }
    }
}

impl Segment {
    pub fn describe(&self) -> String {
        format!("{}{}", self.from, self.to)
    }
}

/// Where a fact came from: the problem statement, a named rule applied to
/// named inputs, or a construction. Every proof step is checkable from this.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Provenance {
    /// Stated by the problem: an assumption, not something proved.
    Given,
    Derived {
        rule: String,
        inputs: Vec<String>,
    },
    Constructed {
        op: String,
    },
    /// Placed by an exact solver -- a point positioned so that a premise holds,
    /// as the falsifier does. Marked apart because *why* a point sits where it
    /// sits changes what may be concluded from it: a solved configuration is a
    /// witness against a claim, never a proof of one.
    Solved {
        op: String,
    },
}

/// `1.0`, the serde default for a fact that was established rather than graded.
fn certain() -> f64 {
    1.0
}

/// A fact in the scene graph: the statement, where it came from, what it
/// depends on, and how much of it is decision versus evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    pub constraint: Constraint,
    pub provenance: Provenance,
    /// Audit fix 1: the dependencies of a fact, named. `provenance` says which
    /// rule fired; `depends_on` names the statements it consumed, so a reader
    /// can walk any conclusion back to what the problem actually gave, and so a
    /// verifier can insist that every dependency is in the ledger.
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// `1.0` means established: the exact predicate holds and, for a derived
    /// fact, a rule replays on the stored graph. Anything lower is a grade from
    /// a grader or a diagram reader -- carried along every later step, and never
    /// quietly rounded up to the truth.
    #[serde(default = "certain")]
    pub confidence: f64,
}

impl Fact {
    /// A statement the problem gave: exact by stipulation, and an assumption
    /// rather than a result.
    pub fn given(constraint: Constraint) -> Self {
        Self {
            constraint,
            provenance: Provenance::Given,
            depends_on: Vec::new(),
            confidence: 1.0,
        }
    }

    /// A fact a rule established, naming the rule and the facts it consumed.
    pub fn derived(constraint: Constraint, rule: &str, depends_on: Vec<String>) -> Self {
        let inputs = depends_on.clone();
        Self {
            constraint,
            provenance: Provenance::Derived {
                rule: rule.to_string(),
                inputs,
            },
            depends_on,
            confidence: 1.0,
        }
    }

    /// A fact observed rather than proved -- a diagram grader's grade of a
    /// stroke or a raster's verdict on a junction. Below `1.0` by construction,
    /// because nothing a picture suggests deserves the kernel's certainty
    /// until a rule replays it.
    pub fn observed(constraint: Constraint, source: &str, confidence: f64) -> Self {
        Self {
            constraint,
            provenance: Provenance::Constructed {
                op: source.to_string(),
            },
            depends_on: Vec::new(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Whether this fact is established, as opposed to merely supported.
    pub fn is_established(&self) -> bool {
        self.confidence >= 1.0
    }

    /// A one-line identity for a fact, used as the name other facts depend on.
    pub fn key(&self) -> String {
        self.constraint.describe()
    }
}

/// `euclidean`, the serde default for a scene that says nothing about its
/// geometry.
fn euclidean_by_default() -> String {
    "euclidean".to_string()
}

/// The canonical geometry IR: objects with identities, facts with provenance
/// and dependencies, the assumption ledger, and the name of the geometry itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneGraph {
    pub points: Vec<KPoint>,
    pub facts: Vec<Fact>,
    /// Audit fix 1: circles are objects, so they live here -- a name, a centre
    /// and an exact squared radius -- instead of being guessed from a picture.
    #[serde(default)]
    pub circles: Vec<KCircle>,
    /// Which geometry the scene claims to live in: `"euclidean"` unless a
    /// statement says otherwise. A string, deliberately, because deciding what
    /// counts as a valid geometry is `geomnonclid`'s job and the kernel refuses
    /// to be the place that quietly assumes an answer.
    #[serde(default = "euclidean_by_default")]
    pub geometry: String,
    /// Statements the problem leaves open. The ledger keeps them visible
    /// instead of silently assumed (the "not established" list).
    pub not_established: Vec<String>,
}

impl SceneGraph {
    pub fn point(&self, name: &str) -> anyhow::Result<&KPoint> {
        self.points
            .iter()
            .find(|p| p.name == name)
            .ok_or_else(|| anyhow::anyhow!("unknown point '{name}'"))
    }

    pub fn coords(&self, name: &str) -> anyhow::Result<(Q, Q)> {
        let p = self.point(name)?;
        Ok((p.x.to_q()?, p.y.to_q()?))
    }

    pub fn has_fact(&self, constraint: &Constraint) -> bool {
        self.facts.iter().any(|f| &f.constraint == constraint)
    }

    /// Record a statement, subject to the kernel's semantics (audit fix 4).
    /// `true` means it entered the ledger.
    ///
    /// Three outcomes, deliberately: the exact predicate accepts it, so it is
    /// recorded; the exact predicate *rejects* it, so it is refused and the
    /// reason goes into the assumption ledger; or the kernel has no semantics for
    /// it yet -- an equality of inscribed angles before the circle machinery
    /// reaches it, a mention of a point nobody declared -- and it is recorded as
    /// an assumption with the gap named in the ledger. Silently accepting and
    /// silently dropping are both ways of losing an audit trail, which is what
    /// this whole exercise was about.
    pub fn add_fact(&mut self, constraint: Constraint, provenance: Provenance) -> bool {
        if self.has_fact(&constraint) {
            return false;
        }
        // `provenance` already names the rule (and travels into the Fact), so
        // only the premise list is read out here.
        let depends_on = match &provenance {
            Provenance::Derived { inputs, .. } => inputs.clone(),
            _ => Vec::new(),
        };
        match self.holds(&constraint) {
            Ok(true) => {}
            Ok(false) => {
                self.record_refusal(format!(
                    "{} is false of the stated figure; the exact predicate rejects it",
                    constraint.describe()
                ));
                return false;
            }
            Err(why) => self.record_refusal(format!(
                "{} is undecidable for the kernel: {why}",
                constraint.describe()
            )),
        }
        let confidence = match &provenance {
            Provenance::Constructed { op } => op
                .strip_prefix("observed:")
                .and_then(|rest| rest.parse::<f64>().ok())
                .map_or(1.0, |grade| grade.clamp(0.0, 1.0)),
            _ => 1.0,
        };
        self.facts.push(Fact {
            constraint,
            provenance,
            depends_on,
            confidence,
        });
        true
    }

    /// Record a conclusion a rule established. Refuses a derivation that names no
    /// premises (audit fix 4): a theorem that depends on nothing is either a
    /// tautology or a bug, and the kernel is not the place to find out which.
    pub fn add_derived(
        &mut self,
        rule: &str,
        depends_on: Vec<String>,
        constraint: Constraint,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(
            !depends_on.is_empty(),
            "{}",
            GeometryError::Unproven(format!(
                "{rule} concluded {} without naming a premise",
                constraint.describe()
            ))
        );
        Ok(self.add_fact(
            constraint,
            Provenance::Derived {
                rule: rule.to_string(),
                inputs: depends_on,
            },
        ))
    }

    /// Write down why the ledger refused or could not decide something, without
    /// repeats.
    fn record_refusal(&mut self, note: String) {
        let line = format!("refused: {note}");
        if !self.not_established.contains(&line) {
            self.not_established.push(line);
        }
    }

    /// The points as the flat coordinate table the standalone predicate helpers
    /// take, sorted by name so the table is the same every time it is built.
    pub fn coord_table(&self) -> anyhow::Result<Vec<(String, Q, Q)>> {
        let mut table = Vec::with_capacity(self.points.len());
        for p in &self.points {
            table.push((p.name.clone(), p.x.to_q()?, p.y.to_q()?));
        }
        table.sort_by(|left, right| left.0.cmp(&right.0));
        Ok(table)
    }

    /// Evaluate a statement in this scene: its points and its circles.
    pub fn holds(&self, constraint: &Constraint) -> anyhow::Result<bool> {
        constraint_holds_in(&self.circles, &self.coord_table()?, constraint)
    }

    pub fn segment(&self, seg: &Segment) -> anyhow::Result<(Q, Q, Q, Q)> {
        let (x1, y1) = self.coords(&seg.from)?;
        let (x2, y2) = self.coords(&seg.to)?;
        Ok((x1, y1, x2, y2))
    }

    /// The direction vector of a segment, refusing degenerate segments (the
    /// two endpoints coincide) instead of propagating a zero direction.
    pub fn direction(&self, seg: &Segment) -> anyhow::Result<(Q, Q)> {
        let (x1, y1, x2, y2) = self.segment(seg)?;
        let dx = x2.sub(&x1)?;
        let dy = y2.sub(&y1)?;
        anyhow::ensure!(
            !dx.is_zero() || !dy.is_zero(),
            "segment {}{} is degenerate: its endpoints coincide",
            seg.from,
            seg.to
        );
        Ok((dx, dy))
    }

    pub fn squared_length(&self, seg: &Segment) -> anyhow::Result<Q> {
        let (dx, dy) = self.direction(seg)?;
        dx.mul(&dx)?.add(&dy.mul(&dy)?)
    }

    /// Declare a point. Refuses a duplicate name and refuses a location already
    /// occupied by another point (audit fix 11): fifteen names is not fifteen
    /// points, and a kernel that accepts `A == B` goes on to divide by `B - A`.
    pub fn add_point(&mut self, point: KPoint) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.points.iter().any(|p| p.name == point.name),
            "{}",
            GeometryError::RepeatedPoint {
                name: point.name.clone()
            }
        );
        let taken = self
            .points
            .iter()
            .find(|p| p.x == point.x && p.y == point.y)
            .map(|p| p.name.clone());
        if let Some(other) = taken {
            return Err(GeometryError::CoincidentPoints {
                a: other,
                b: point.name,
            }
            .into());
        }
        self.points.push(point);
        Ok(())
    }

    /// Declare a circle (audit fix 1): a named object whose centre must be a
    /// declared point and whose squared radius must be a positive rational.
    pub fn add_circle(&mut self, circle: KCircle) -> anyhow::Result<()> {
        self.point(&circle.center)?;
        anyhow::ensure!(
            !self.circles.iter().any(|c| c.name == circle.name),
            "{}",
            GeometryError::RepeatedPoint {
                name: circle.name.clone()
            }
        );
        let radius = circle.radius_sq.to_q()?;
        anyhow::ensure!(
            Q::ZERO.less(&radius),
            "{}",
            GeometryError::ZeroRadiusCircle {
                circle: circle.name.clone()
            }
        );
        self.circles.push(circle);
        Ok(())
    }

    pub fn circle(&self, name: &str) -> anyhow::Result<&KCircle> {
        self.circles
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| GeometryError::UnknownCircle(name.to_string()).into())
    }

    pub fn has_circle(&self, name: &str) -> bool {
        self.circles.iter().any(|c| c.name == name)
    }

    /// The exact angle data at a named angle: rational dot and cross products and
    /// rational squared norms. This is the only angle arithmetic the kernel
    /// permits, and it is why `AngleIs` names a cosine rather than a number of
    /// degrees (audit fix 5).
    pub fn angle_terms(&self, angle: &Angle3) -> anyhow::Result<AngleTerms> {
        let at = self.coords(&angle.at)?;
        let from = self.coords(&angle.from)?;
        let to = self.coords(&angle.to)?;
        AngleTerms::from_coords(at, from, to)
    }

    /// The statements taken as given. Not everything in the ledger is an
    /// assumption, and a checker that cannot tell assumptions from conclusions
    /// cannot tell a proof from a wish.
    pub fn assumptions(&self) -> Vec<&Fact> {
        self.facts
            .iter()
            .filter(|f| matches!(f.provenance, Provenance::Given))
            .collect()
    }

    /// The statements established rather than given, each naming its rule.
    pub fn derived(&self) -> Vec<&Fact> {
        self.facts
            .iter()
            .filter(|f| !matches!(f.provenance, Provenance::Given))
            .collect()
    }

    /// The confidence of the weakest support a claim rests on, walked along its
    /// dependency chain: the product of the confidences, which audit fix 10
    /// demands be carried rather than dropped. A conclusion resting on a fact a
    /// diagram grader graded `0.83` is worth `0.83`, however many exact rules
    /// stand between them.
    pub fn confidence_in(&self, target: &Constraint) -> anyhow::Result<f64> {
        let fact = self
            .facts
            .iter()
            .find(|f| &f.constraint == target)
            .ok_or_else(|| {
                GeometryError::Unproven(format!("{} is not in the ledger", target.describe()))
            })?;
        let mut seen: Vec<String> = Vec::new();
        let mut chain = 1.0f64;
        let mut queue: Vec<&Fact> = vec![fact];
        while let Some(current) = queue.pop() {
            let key = current.key();
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            chain *= current.confidence;
            for dependency in &current.depends_on {
                if let Some(next) = self.facts.iter().find(|f| f.key() == *dependency) {
                    queue.push(next);
                }
            }
        }
        Ok(chain)
    }

    /// Every point the scene mentions by name -- points, circle centres -- so a
    /// falsifier can enumerate what it has to place without inventing an
    /// indexing scheme of its own.
    pub fn all_point_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.points.iter().map(|p| p.name.clone()).collect();
        for circle in &self.circles {
            if !names.contains(&circle.center) {
                names.push(circle.center.clone());
            }
        }
        names
    }

    pub fn describe_facts(&self) -> String {
        self.facts
            .iter()
            .map(|f| {
                let source = match &f.provenance {
                    Provenance::Given => "given".to_string(),
                    Provenance::Derived { rule, inputs } => {
                        format!("by {rule} from {}", inputs.join("+"))
                    }
                    Provenance::Constructed { op } => format!("by construction {op}"),
                    Provenance::Solved { op } => format!("by solving {op}"),
                };
                let mut line = format!("{} [{source}]", f.constraint.describe());
                if !f.depends_on.is_empty() && !matches!(f.provenance, Provenance::Derived { .. }) {
                    line.push_str(&format!(" depends on {}", f.depends_on.join("+")));
                }
                if !f.is_established() {
                    line.push_str(&format!(" (confidence {:.2})", f.confidence));
                }
                line
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// The kernel's semantics: the one place a predicate's *meaning* is defined,
/// shared by the rules, the proof verifier and the counterexample sampler so
/// that none of them can disagree about what a claim says. Every arm is exact.
///
/// Audit fixes 4 and 11 together decide the shape of this function: a predicate
/// whose objects have gone degenerate is `false` or an error, never `true` by
/// default. A collapsed segment, a triangle with no area, an angle with a leg of
/// zero length -- none of them get to hold vacuously, because "true by default"
/// is precisely how an empty derivation becomes a claimed theorem.
pub fn constraint_holds_in(
    circles: &[KCircle],
    coords: &[(String, Q, Q)],
    constraint: &Constraint,
) -> anyhow::Result<bool> {
    let xy = |name: &str| -> anyhow::Result<(Q, Q)> {
        coords
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, x, y)| (*x, *y))
            .ok_or_else(|| anyhow::anyhow!("unknown point '{name}'"))
    };
    let find_circle = |name: &str| -> anyhow::Result<&KCircle> {
        circles
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| anyhow::Error::from(GeometryError::UnknownCircle(name.to_string())))
    };
    let dist2 = |p: (Q, Q), q: (Q, Q)| -> anyhow::Result<Q> {
        let dx = p.0.sub(&q.0)?;
        let dy = p.1.sub(&q.1)?;
        dx.mul(&dx)?.add(&dy.mul(&dy)?)
    };
    let angle_of = |a: &Angle3| -> anyhow::Result<AngleTerms> {
        AngleTerms::from_coords(xy(&a.at)?, xy(&a.from)?, xy(&a.to)?)
    };
    match constraint {
        Constraint::Collinear { a, b, c } => {
            let (pa, pb, pc) = (xy(a)?, xy(b)?, xy(c)?);
            let u = pb.0.sub(&pa.0)?;
            let v = pb.1.sub(&pa.1)?;
            let w = pc.0.sub(&pa.0)?;
            let z = pc.1.sub(&pa.1)?;
            Ok(u.mul(&z)?.sub(&v.mul(&w)?)?.is_zero())
        }
        Constraint::Parallel { first, second } => {
            let ((ax, ay), (bx, by)) = (xy(&first.from)?, xy(&first.to)?);
            let ((cx, cy), (dx, dy)) = (xy(&second.from)?, xy(&second.to)?);
            let (u, v) = (bx.sub(&ax)?, by.sub(&ay)?);
            let (w, z) = (dx.sub(&cx)?, dy.sub(&cy)?);
            anyhow::ensure!(!u.is_zero() || !v.is_zero(), "degenerate first segment");
            anyhow::ensure!(!w.is_zero() || !z.is_zero(), "degenerate second segment");
            Ok(u.mul(&z)?.sub(&v.mul(&w)?)?.is_zero())
        }
        Constraint::Perpendicular { first, second } => {
            let ((ax, ay), (bx, by)) = (xy(&first.from)?, xy(&first.to)?);
            let ((cx, cy), (dx, dy)) = (xy(&second.from)?, xy(&second.to)?);
            let (u, v) = (bx.sub(&ax)?, by.sub(&ay)?);
            let (w, z) = (dx.sub(&cx)?, dy.sub(&cy)?);
            anyhow::ensure!(!u.is_zero() || !v.is_zero(), "degenerate first segment");
            anyhow::ensure!(!w.is_zero() || !z.is_zero(), "degenerate second segment");
            Ok(u.mul(&w)?.add(&v.mul(&z)?)?.is_zero())
        }
        Constraint::EqualLength { first, second } => {
            let ((ax, ay), (bx, by)) = (xy(&first.from)?, xy(&first.to)?);
            let ((cx, cy), (dx, dy)) = (xy(&second.from)?, xy(&second.to)?);
            let ux = bx.sub(&ax)?;
            let uy = by.sub(&ay)?;
            let wx = dx.sub(&cx)?;
            let wy = dy.sub(&cy)?;
            let l1 = ux.mul(&ux)?.add(&uy.mul(&uy)?)?;
            let l2 = wx.mul(&wx)?.add(&wy.mul(&wy)?)?;
            Ok(l1 == l2)
        }
        Constraint::MidpointOf { p, a, b } => {
            let (pp, pa, pb) = (xy(p)?, xy(a)?, xy(b)?);
            let mx = pa.0.add(&pb.0)?.half()?;
            let my = pa.1.add(&pb.1)?.half()?;
            Ok(pp.0 == mx && pp.1 == my)
        }
        Constraint::Distinct { a, b } => {
            let (pa, pb) = (xy(a)?, xy(b)?);
            Ok(pa != pb)
        }
        Constraint::NonCollinear { a, b, c } | Constraint::Triangle { a, b, c } => {
            // Audit fix 4's precondition, made checkable: a triangle is a
            // triangle because its doubled area is a nonzero rational.
            let (pa, pb, pc) = (xy(a)?, xy(b)?, xy(c)?);
            Ok(!cross2(pa, pb, pc)?.is_zero())
        }
        Constraint::Between { a, m, b } => {
            let (pa, pm, pb) = (xy(a)?, xy(m)?, xy(b)?);
            if !cross2(pa, pm, pb)?.is_zero() {
                return Ok(false);
            }
            // Collinear, so two positive dot products pin `m` strictly inside:
            // not at an end, not outside, not on top of either point.
            let t = (pb.0.sub(&pa.0)?, pb.1.sub(&pa.1)?);
            let u = (pm.0.sub(&pa.0)?, pm.1.sub(&pa.1)?);
            let v = (pb.0.sub(&pm.0)?, pb.1.sub(&pm.1)?);
            let along = u.0.mul(&t.0)?.add(&u.1.mul(&t.1)?)?;
            let back = v.0.mul(&t.0)?.add(&v.1.mul(&t.1)?)?;
            Ok(Q::ZERO.less(&along) && Q::ZERO.less(&back))
        }
        Constraint::RatioOf { p, a, b, num, den } => {
            anyhow::ensure!(
                *num > 0 && *den > 0,
                GeometryError::EmptyGeometry(format!(
                    "the ratio {num}:{den} does not divide a segment into two positive parts"
                ))
            );
            let (pp, pa, pb) = (xy(p)?, xy(a)?, xy(b)?);
            let (n, d) = (Q::from_int(*num), Q::from_int(*den));
            let total = d.add(&n)?;
            let want_x = d.mul(&pa.0)?.add(&n.mul(&pb.0)?)?.div(&total)?;
            let want_y = d.mul(&pa.1)?.add(&n.mul(&pb.1)?)?.div(&total)?;
            Ok(pp.0 == want_x && pp.1 == want_y)
        }
        Constraint::LengthIs { seg, square } => {
            let (pa, pb) = (xy(&seg.from)?, xy(&seg.to)?);
            Ok(dist2(pa, pb)? == square.to_q()?)
        }
        Constraint::ScaleLength {
            first,
            second,
            num,
            den,
        } => {
            anyhow::ensure!(
                *num > 0 && *den > 0,
                GeometryError::EmptyGeometry(format!(
                    "the scale {num}/{den} is not a positive ratio"
                ))
            );
            let (fa, fb) = (xy(&first.from)?, xy(&first.to)?);
            let (sa, sb) = (xy(&second.from)?, xy(&second.to)?);
            let (n, d) = (Q::from_int(*num), Q::from_int(*den));
            // `|second| == (n/d)|first|`, squared so the test stays rational:
            // d^2 |second|^2 == n^2 |first|^2.
            Ok(d.mul(&d)?.mul(&dist2(sa, sb)?)? == n.mul(&n)?.mul(&dist2(fa, fb)?)?)
        }
        Constraint::AngleEqual { first, second } => angle_of(first)?.equal(&angle_of(second)?),
        Constraint::RightAngle { at } => Ok(angle_of(at)?.is_right()),
        Constraint::AngleIs { at, cos } => Ok(angle_of(at)?.cosine()? == *cos),
        Constraint::AreaEqual { first, second } => {
            Ok(doubled_area_magnitude(coords, first)? == doubled_area_magnitude(coords, second)?)
        }
        Constraint::Congruent { first, second } => Ok(sides_match(
            &sides_sq(coords, first)?,
            &sides_sq(coords, second)?,
        )?),
        Constraint::Circle {
            name,
            center,
            radius_sq,
        } => {
            // A circle predicate holds when the circle *object* carrying that
            // identity has exactly these parameters. An object is not
            // interchangeable with a look-alike, which is the entire point of
            // handing it an identity in the first place.
            let found = find_circle(name)?;
            Ok(&found.center == center && found.radius_sq == *radius_sq)
        }
        Constraint::OnCircle { p, circle } => {
            let found = find_circle(circle)?;
            let (pp, pc) = (xy(p)?, xy(&found.center)?);
            Ok(dist2(pp, pc)? == found.radius_sq.to_q()?)
        }
        Constraint::Diameter { circle, a, b } => {
            let found = find_circle(circle)?;
            let (pa, pb, pc) = (xy(a)?, xy(b)?, xy(&found.center)?);
            let radius = found.radius_sq.to_q()?;
            let mid_x = pa.0.add(&pb.0)?.half()?;
            let mid_y = pa.1.add(&pb.1)?.half()?;
            Ok(dist2(pa, pc)? == radius
                && dist2(pb, pc)? == radius
                && pc.0 == mid_x
                && pc.1 == mid_y)
        }
    }
}

/// The predicate's meaning over points alone. The circle predicates need the
/// scene's circle objects and therefore refuse here: `constraint_holds_in` is
/// the general form, and a caller holding a scene graph should use it.
pub fn constraint_holds(
    coords: &[(String, Q, Q)],
    constraint: &Constraint,
) -> anyhow::Result<bool> {
    constraint_holds_in(&[], coords, constraint)
}

/// `(b - a) x (c - a)`: twice the signed area of `abc`, and therefore the
/// collinearity test and the side-of-line test in one exact number. Rules use
/// it as their side condition -- which side of a transversal a ray lies on is a
/// sign, not a judgement call.
pub fn cross2(a: (Q, Q), b: (Q, Q), c: (Q, Q)) -> anyhow::Result<Q> {
    b.0.sub(&a.0)?
        .mul(&c.1.sub(&a.1)?)?
        .sub(&b.1.sub(&a.1)?.mul(&c.0.sub(&a.0)?)?)
}

/// The signed doubled area of a named triangle.
pub fn doubled_area(coords: &[(String, Q, Q)], tri: &Tri3) -> anyhow::Result<Q> {
    let (pa, pb, pc) = (
        point_xy(coords, &tri.a)?,
        point_xy(coords, &tri.b)?,
        point_xy(coords, &tri.c)?,
    );
    cross2(pa, pb, pc)
}

/// The unsigned doubled area, the quantity an area comparison uses.
pub fn doubled_area_magnitude(coords: &[(String, Q, Q)], tri: &Tri3) -> anyhow::Result<Q> {
    let area = doubled_area(coords, tri)?;
    Ok(if area.less(&Q::ZERO) {
        area.neg()?
    } else {
        area
    })
}

/// Look one point up in a coordinate table.
pub fn point_xy(coords: &[(String, Q, Q)], name: &str) -> anyhow::Result<(Q, Q)> {
    coords
        .iter()
        .find(|(n, _, _)| n == name)
        .map(|(_, x, y)| (*x, *y))
        .ok_or_else(|| anyhow::anyhow!("unknown point '{name}'"))
}

/// The three squared side lengths of a named triangle, in the order `ab`, `bc`,
/// `ca`. Squared, because a lattice triangle's side lengths are usually
/// irrational while their squares never are.
pub fn sides_sq(coords: &[(String, Q, Q)], tri: &Tri3) -> anyhow::Result<[Q; 3]> {
    let (pa, pb, pc) = (
        point_xy(coords, &tri.a)?,
        point_xy(coords, &tri.b)?,
        point_xy(coords, &tri.c)?,
    );
    let norm = |p: (Q, Q), q: (Q, Q)| -> anyhow::Result<Q> {
        let dx = p.0.sub(&q.0)?;
        let dy = p.1.sub(&q.1)?;
        dx.mul(&dx)?.add(&dy.mul(&dy)?)
    };
    Ok([norm(pa, pb)?, norm(pb, pc)?, norm(pc, pa)?])
}

/// Equality of two multisets of three exact squared lengths, without asking a
/// rational to order itself: `Q` carries no `Ord`, and inventing one just to
/// sort would be exactly the kind of quiet convenience the audit was about.
fn sides_match(first: &[Q; 3], second: &[Q; 3]) -> anyhow::Result<bool> {
    let mut taken = [false; 3];
    for want in first {
        let mut matched = false;
        for index in 0..3 {
            if !taken[index] && *want == second[index] {
                taken[index] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Non-degeneracy errors every construction can refuse with. Not `Copy`: the
/// useful variants carry the names and vertices that went wrong, because an
/// error that cannot say which of the fifteen points collapsed is an error a
/// caller cannot act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeometryError {
    ParallelLines,
    CoincidentLines,
    DegenerateSegment,
    /// Audit fix 11: two names for one location. The old kernel accepted a
    /// scene in which `A` and `B` had the same coordinates and then divided by
    /// `B - A`; the audit found results resting on exactly that.
    CoincidentPoints {
        a: String,
        b: String,
    },
    /// A name used twice for two different locations.
    RepeatedPoint {
        name: String,
    },
    /// Audit fix 4: a triangle with no area, which every triangle theorem in
    /// the library silently assumes away.
    DegenerateTriangle {
        vertices: String,
    },
    /// An angle with no legs, or with a leg of zero length.
    DegenerateAngle {
        detail: String,
    },
    ZeroRadiusCircle {
        circle: String,
    },
    PointNotOnCircle {
        point: String,
        circle: String,
    },
    UnknownCircle(String),
    /// Audit fix 7 seen from the other side: some constructions genuinely need
    /// square roots -- a tangent point, a circle meeting a line -- and the
    /// rational kernel says so rather than returning a float dressed as an
    /// answer.
    IrrationalRequired(String),
    /// An exact solve that has no rational solution at all.
    NoRationalSolution(String),
    /// A scene with nothing in it, or a claim about nothing.
    EmptyGeometry(String),
    /// Something asserted as established that no rule established: an
    /// assumption smuggled through as a fact, or a fact whose stated
    /// dependencies are missing from the ledger.
    Unproven(String),
    ArithmeticOverflow,
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParallelLines => write!(f, "the lines are parallel and do not intersect"),
            Self::CoincidentLines => write!(f, "the lines are coincident"),
            Self::DegenerateSegment => write!(f, "a segment's endpoints coincide"),
            Self::CoincidentPoints { a, b } => {
                write!(f, "'{a}' and '{b}' name the same location")
            }
            Self::RepeatedPoint { name } => write!(f, "'{name}' is declared twice"),
            Self::DegenerateTriangle { vertices } => {
                write!(
                    f,
                    "the triangle {vertices} is degenerate: no area to reason with"
                )
            }
            Self::DegenerateAngle { detail } => write!(f, "degenerate angle: {detail}"),
            Self::ZeroRadiusCircle { circle } => {
                write!(f, "the circle '{circle}' has zero radius")
            }
            Self::PointNotOnCircle { point, circle } => {
                write!(f, "'{point}' is not on the circle '{circle}'")
            }
            Self::UnknownCircle(name) => write!(f, "no circle named '{name}' in the scene"),
            Self::IrrationalRequired(detail) => {
                write!(
                    f,
                    "the exact answer needs an irrational coordinate: {detail}"
                )
            }
            Self::NoRationalSolution(detail) => write!(f, "no exact rational solution: {detail}"),
            Self::EmptyGeometry(detail) => {
                write!(f, "there is no geometry to reason about: {detail}")
            }
            Self::Unproven(detail) => write!(f, "not established by any rule: {detail}"),
            Self::ArithmeticOverflow => write!(f, "exact integer arithmetic overflowed"),
        }
    }
}

impl std::error::Error for GeometryError {}

/// The exact data of an angle: the dot and the 2D cross product of its two
/// legs, plus the two squared leg norms. Every angle relation the kernel
/// states -- equal, right, supplementary, equal to a named exact cosine -- is
/// decidable on these four rationals, which is why exact angles never needed
/// the `arccos` whose absence the audit listed as a missing capability. The
/// capability was never the inverse cosine; it was comparing one exact quantity
/// with another.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AngleTerms {
    pub dot: Q,
    pub cross: Q,
    pub nu: Q,
    pub nv: Q,
}

/// `num_a / sqrt(norm_a) == num_b / sqrt(norm_b)`, exactly, sign included.
/// Both norms are positive, so the two sides agree exactly when they carry the
/// same sign and their squares agree -- a test over rationals only.
fn radical_ratio_equal(num_a: &Q, norm_a: &Q, num_b: &Q, norm_b: &Q) -> anyhow::Result<bool> {
    let (zero_a, zero_b) = (num_a.is_zero(), num_b.is_zero());
    if zero_a != zero_b {
        return Ok(false);
    }
    if zero_a {
        return Ok(true);
    }
    if num_a.less(&Q::ZERO) != num_b.less(&Q::ZERO) {
        return Ok(false);
    }
    Ok(num_a.mul(num_a)?.mul(norm_b)? == num_b.mul(num_b)?.mul(norm_a)?)
}

/// `value == square^2 * square_free`, `square` a positive rational and
/// `square_free` a square-free integer `>= 1`. Written as
/// `num / den == (num * den) / den^2` so only one integer needs splitting.
fn squarefree_rational(value: &Q) -> anyhow::Result<(Q, i64)> {
    anyhow::ensure!(
        Q::ZERO.less(value),
        GeometryError::DegenerateAngle {
            detail: format!("a product of squared norms must be positive, got {value}"),
        }
    );
    let num = i64::try_from(value.num).map_err(|_| GeometryError::ArithmeticOverflow)?;
    let den = i64::try_from(value.den).map_err(|_| GeometryError::ArithmeticOverflow)?;
    let (num_square, num_free) = squarefree_split(num)?;
    let (den_square, den_free) = squarefree_split(den)?;
    let free = num_free
        .checked_mul(den_free)
        .ok_or(GeometryError::ArithmeticOverflow)?;
    let (extra, square_free) = squarefree_split(free)?;
    let top = Q::from_int(
        num_square
            .checked_mul(extra)
            .ok_or(GeometryError::ArithmeticOverflow)?,
    );
    let bottom = Q::from_int(
        den_square
            .checked_mul(den_free)
            .ok_or(GeometryError::ArithmeticOverflow)?,
    );
    Ok((top.div(&bottom)?, square_free))
}

impl AngleTerms {
    /// The terms of the angle at `at`, between the legs to `from` and to `to`.
    /// A zero-length leg is not an angle at all -- audit fix 11 applied here, at
    /// the place where the old kernel would simply have gone on computing.
    pub fn from_coords(at: (Q, Q), from: (Q, Q), to: (Q, Q)) -> anyhow::Result<Self> {
        let u = (from.0.sub(&at.0)?, from.1.sub(&at.1)?);
        let v = (to.0.sub(&at.0)?, to.1.sub(&at.1)?);
        let nu = u.0.mul(&u.0)?.add(&u.1.mul(&u.1)?)?;
        let nv = v.0.mul(&v.0)?.add(&v.1.mul(&v.1)?)?;
        anyhow::ensure!(
            Q::ZERO.less(&nu) && Q::ZERO.less(&nv),
            GeometryError::DegenerateAngle {
                detail: "one leg has zero length".to_string(),
            }
        );
        Ok(Self {
            dot: u.0.mul(&v.0)?.add(&u.1.mul(&v.1)?)?,
            cross: u.0.mul(&v.1)?.sub(&u.1.mul(&v.0)?)?,
            nu,
            nv,
        })
    }

    fn norms(&self) -> anyhow::Result<Q> {
        let norm = self.nu.mul(&self.nv)?;
        anyhow::ensure!(
            Q::ZERO.less(&norm),
            GeometryError::DegenerateAngle {
                detail: "zero-length leg".to_string()
            }
        );
        Ok(norm)
    }

    /// Angle equality, exact. Two angles in `[0, pi]` are equal exactly when
    /// their cosines are, and comparing cosines touches no float.
    pub fn equal(&self, other: &Self) -> anyhow::Result<bool> {
        radical_ratio_equal(&self.dot, &self.norms()?, &other.dot, &other.norms()?)
    }

    /// Supplementary angles: equal and opposite cosines, which is the relation
    /// a transversal across parallel lines produces, and the one an angle chase
    /// most often needs.
    pub fn supplementary(&self, other: &Self) -> anyhow::Result<bool> {
        let opposite = self.dot.neg()?;
        radical_ratio_equal(&opposite, &self.norms()?, &other.dot, &other.norms()?)
    }

    /// A right angle is a zero dot product, and a zero dot product is a rational
    /// that is zero: a decision, not a tolerance.
    pub fn is_right(&self) -> bool {
        self.dot.is_zero()
    }

    pub fn is_obtuse(&self) -> anyhow::Result<bool> {
        Ok(self.dot.less(&Q::ZERO))
    }

    /// The exact cosine, an element of `Q(sqrt d)`.
    pub fn cosine(&self) -> anyhow::Result<QSqrt> {
        self.under_root(&self.dot)
    }

    /// The exact signed sine, `cross / sqrt(|u|^2 |v|^2)` -- signed, so a
    /// reader that cares about orientation can use it.
    pub fn sine(&self) -> anyhow::Result<QSqrt> {
        self.under_root(&self.cross)
    }

    fn under_root(&self, numerator: &Q) -> anyhow::Result<QSqrt> {
        let (square, square_free) = squarefree_rational(&self.norms()?)?;
        if square_free == 1 {
            return Ok(QSqrt::rational(Frac::from_q(numerator.div(&square)?)));
        }
        let coefficient = numerator.div(&square)?.div(&Q::from_int(square_free))?;
        QSqrt::new(Frac::from_int(0), Frac::from_q(coefficient), square_free)
    }

    /// Numeric evidence for a report or a diagram label, never a fact.
    pub fn degrees_estimate(&self) -> f64 {
        let degrees = self.cross.to_f64().atan2(self.dot.to_f64()).to_degrees();
        if degrees < 0.0 {
            degrees + 360.0
        } else {
            degrees
        }
    }
}

/// The typed construction grammar. Every operation is exact and refuses
/// invalid configurations instead of propagating them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Construction {
    Midpoint {
        a: String,
        b: String,
    },
    Intersection {
        first: Segment,
        second: Segment,
    },
    /// The foot of the perpendicular from `p` to the line `ab`.
    Projection {
        p: String,
        a: String,
        b: String,
    },
    /// The mirror of `p` across the line `ab`.
    Reflect {
        p: String,
        a: String,
        b: String,
    },
}

impl Construction {
    pub fn describe(&self) -> String {
        match self {
            Self::Midpoint { a, b } => format!("midpoint({a},{b})"),
            Self::Intersection { first, second } => {
                format!(
                    "intersection({}{},{}{})",
                    first.from, first.to, second.from, second.to
                )
            }
            Self::Projection { p, a, b } => format!("projection({p},{a}{b})"),
            Self::Reflect { p, a, b } => format!("reflect({p},{a}{b})"),
        }
    }
}

/// Apply a construction exactly, returning the new point.
pub fn construct(graph: &SceneGraph, op: &Construction, name: &str) -> anyhow::Result<KPoint> {
    let coords_of = |n: &str| graph.coords(n);
    match op {
        Construction::Midpoint { a, b } => {
            let (pa, pb) = (coords_of(a)?, coords_of(b)?);
            Ok(KPoint {
                name: name.to_string(),
                x: Frac::from_q(pa.0.add(&pb.0)?.half()?),
                y: Frac::from_q(pa.1.add(&pb.1)?.half()?),
            })
        }
        Construction::Intersection { first, second } => {
            let (x1, y1) = coords_of(&first.from)?;
            let (x2, y2) = coords_of(&first.to)?;
            let (x3, y3) = coords_of(&second.from)?;
            let (x4, y4) = coords_of(&second.to)?;
            let (u, v) = (x2.sub(&x1)?, y2.sub(&y1)?);
            let (w, z) = (x4.sub(&x3)?, y4.sub(&y3)?);
            anyhow::ensure!(
                !u.is_zero() || !v.is_zero(),
                GeometryError::DegenerateSegment
            );
            anyhow::ensure!(
                !w.is_zero() || !z.is_zero(),
                GeometryError::DegenerateSegment
            );
            let denom = u.mul(&z)?.sub(&v.mul(&w)?)?;
            if denom.is_zero() {
                // coincident iff the second line's origin lies on the first
                let cross = x3.sub(&x1)?.mul(&v)?.sub(&y3.sub(&y1)?.mul(&u)?)?;
                if cross.is_zero() {
                    anyhow::bail!(GeometryError::CoincidentLines);
                }
                anyhow::bail!(GeometryError::ParallelLines);
            }
            let t = x3
                .sub(&x1)?
                .mul(&z)?
                .sub(&y3.sub(&y1)?.mul(&w)?)?
                .div(&denom)?;
            let x = x1.add(&u.mul(&t)?)?;
            let y = y1.add(&v.mul(&t)?)?;
            Ok(KPoint {
                name: name.to_string(),
                x: Frac::from_q(x),
                y: Frac::from_q(y),
            })
        }
        Construction::Projection { p, a, b } => {
            let (pp, pa, pb) = (coords_of(p)?, coords_of(a)?, coords_of(b)?);
            let (u, v) = (pb.0.sub(&pa.0)?, pb.1.sub(&pa.1)?);
            anyhow::ensure!(
                !u.is_zero() || !v.is_zero(),
                GeometryError::DegenerateSegment
            );
            let t =
                pp.0.sub(&pa.0)?
                    .mul(&u)?
                    .add(&pp.1.sub(&pa.1)?.mul(&v)?)?
                    .div(&u.mul(&u)?.add(&v.mul(&v)?)?)?;
            let x = pa.0.add(&u.mul(&t)?)?;
            let y = pa.1.add(&v.mul(&t)?)?;
            Ok(KPoint {
                name: name.to_string(),
                x: Frac::from_q(x),
                y: Frac::from_q(y),
            })
        }
        Construction::Reflect { p, a, b } => {
            let foot = construct(
                graph,
                &Construction::Projection {
                    p: p.to_string(),
                    a: a.to_string(),
                    b: b.to_string(),
                },
                name,
            )?;
            let (pp, pf) = (coords_of(p)?, (foot.x.to_q()?, foot.y.to_q()?));
            let x = pf.0.mul(&Q::from_int(2))?.sub(&pp.0)?;
            let y = pf.1.mul(&Q::from_int(2))?.sub(&pp.1)?;
            Ok(KPoint {
                name: name.to_string(),
                x: Frac::from_q(x),
                y: Frac::from_q(y),
            })
        }
    }
}

/// A deduction rule: typed preconditions, checked against the graph, and the
/// conclusions those preconditions license. Applying one emits facts whose
/// provenance names the rule and the facts it consumed -- the machine-checkable
/// certificate. Not `Copy`: the angle rules carry the names they bind over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rule {
    /// midpoint(p, a, b) -> collinear(a, p, b) and len(ap) = len(pb)
    MidpointCollinear,
    /// collinear(a,b,c) and collinear(a,b,d) with a != b -> collinear(a,c,d)
    CollinearTransitivity,
    /// ab parallel cd and cd parallel ef -> ab parallel ef
    ParallelTransitivity,
    /// ab perp cd and cd perp ef -> ab parallel ef
    PerpendicularTransitivity,
    /// ab parallel cd and cd perp ef -> ab perp ef
    ParallelPerpendicular,
    /// the diagonals ac and bd sharing a midpoint make abcd a parallelogram:
    /// ab parallel cd, bc parallel ad, and len(ab) = len(cd)
    ParallelogramDiagonals,
    /// EqualAngles: the inscribed-angle theorem, with the centre angle as a
    /// third conclusion. Its preconditions are the four incidences on the circle
    /// and the fact that the arc the two inscribed angles stand on subtends a
    /// centre angle -- and audit fix 4 is exactly why this rule exists in a
    /// kernel that refuses to state them: the classical "angles subtending the
    /// same arc are equal" is *false* when the two vertices sit on opposite
    /// arcs, where the angles are supplementary rather than equal, so the rule
    /// has to refuse rather than infer the configuration the theorem needs.
    AngleEqual {
        circle: String,
        center: String,
        a: String,
        b: String,
        c: String,
        d: String,
    },
    /// AngleIsosceles: equal sides give equal opposite angles, but only of a
    /// triangle, so `Triangle` is a premise and a degenerate three-point frame
    /// is refused rather than quietly treated as a very flat triangle.
    AngleIsosceles {
        apex: String,
        left: String,
        right: String,
    },
    /// MidpointBisectsPerpendicular: the segment from a vertex to the midpoint
    /// of the opposite side is perpendicular to that side only in an isosceles
    /// triangle; state it as a premise, get the two equal sides back.
    MidpointBisectsPerpendicular {
        apex: String,
        midpoint: String,
        left: String,
        right: String,
    },
    /// EqualAnglesSubtendArc: the converse -- two points on the same side of a
    /// chord that see it under the same angle lie on the circle through that
    /// chord. The same-side test is a sign of an exact cross product, not a
    /// picture of where the points look like they are.
    EqualAnglesSubtendArc {
        circle: String,
        chord_left: String,
        chord_right: String,
        first: String,
        second: String,
    },
}

pub const RULES: [Rule; 10] = [
    Rule::MidpointCollinear,
    Rule::CollinearTransitivity,
    Rule::ParallelTransitivity,
    Rule::PerpendicularTransitivity,
    Rule::ParallelPerpendicular,
    Rule::ParallelogramDiagonals,
    Rule::AngleEqual {
        circle: String::new(),
        center: String::new(),
        a: String::new(),
        b: String::new(),
        c: String::new(),
        d: String::new(),
    },
    Rule::AngleIsosceles {
        apex: String::new(),
        left: String::new(),
        right: String::new(),
    },
    Rule::MidpointBisectsPerpendicular {
        apex: String::new(),
        midpoint: String::new(),
        left: String::new(),
        right: String::new(),
    },
    Rule::EqualAnglesSubtendArc {
        circle: String::new(),
        chord_left: String::new(),
        chord_right: String::new(),
        first: String::new(),
        second: String::new(),
    },
];

impl Rule {
    pub fn name(&self) -> &'static str {
        match self {
            Self::MidpointCollinear => "MidpointCollinear",
            Self::CollinearTransitivity => "CollinearTransitivity",
            Self::ParallelTransitivity => "ParallelTransitivity",
            Self::PerpendicularTransitivity => "PerpendicularTransitivity",
            Self::ParallelPerpendicular => "ParallelPerpendicular",
            Self::ParallelogramDiagonals => "ParallelogramDiagonals",
            Self::AngleEqual { .. } => "AngleEqual",
            Self::AngleIsosceles { .. } => "AngleIsosceles",
            Self::MidpointBisectsPerpendicular { .. } => "MidpointBisectsPerpendicular",
            Self::EqualAnglesSubtendArc { .. } => "EqualAnglesSubtendArc",
        }
    }

    /// Check preconditions against the graph and return every conclusion the
    /// rule licenses, each naming the facts it consumed (empty when the
    /// preconditions fail).
    ///
    /// The `Err` path is reserved for what the scene cannot answer at all -- a
    /// point nobody declared, an angle with a leg of zero length -- because a
    /// rule that cannot tell "the premise is absent" from "the figure is
    /// malformed" teaches the saturation loop nothing about which to fix.
    pub fn forward(&self, graph: &SceneGraph) -> anyhow::Result<Vec<(Constraint, Vec<String>)>> {
        let mut out = Vec::new();
        // Which side of the line `p q` a point `r` falls on, as the sign of an
        // exact cross product. Rules about arcs, betweenness and interior
        // cevians are side conditions, and a side condition decided by eyeballing
        // a diagram is how the audit's false theorems got proved.
        let side = |p: &str, q: &str, r: &str| -> anyhow::Result<i8> {
            let value = cross2(graph.coords(p)?, graph.coords(q)?, graph.coords(r)?)?;
            Ok(if value.less(&Q::ZERO) {
                -1
            } else if value.is_zero() {
                0
            } else {
                1
            })
        };
        match self {
            Self::MidpointCollinear => {
                for fact in &graph.facts {
                    if let Constraint::MidpointOf { p, a, b } = &fact.constraint {
                        out.push((
                            Constraint::Collinear {
                                a: a.clone(),
                                b: p.clone(),
                                c: b.clone(),
                            },
                            vec![fact.constraint.describe()],
                        ));
                        out.push((
                            Constraint::EqualLength {
                                first: Segment {
                                    from: a.clone(),
                                    to: p.clone(),
                                },
                                second: Segment {
                                    from: p.clone(),
                                    to: b.clone(),
                                },
                            },
                            vec![fact.constraint.describe()],
                        ));
                    }
                }
            }
            Self::CollinearTransitivity => {
                // two collinear triples sharing a distinct pair lie on one
                // line: every triple of the union is derivable
                let triples: Vec<(Vec<String>, String)> = graph
                    .facts
                    .iter()
                    .filter_map(|f| match &f.constraint {
                        Constraint::Collinear { a, b, c } => Some((
                            vec![a.clone(), b.clone(), c.clone()],
                            f.constraint.describe(),
                        )),
                        _ => None,
                    })
                    .collect();
                for (i, (left, left_desc)) in triples.iter().enumerate() {
                    for (right, right_desc) in triples.iter().skip(i + 1) {
                        let mut shared: Vec<&String> =
                            left.iter().filter(|p| right.contains(p)).collect();
                        shared.sort();
                        shared.dedup();
                        if shared.len() < 2 {
                            continue;
                        }
                        let mut union = left.clone();
                        for p in right {
                            if !union.contains(p) {
                                union.push(p.clone());
                            }
                        }
                        if union.len() > 5 {
                            continue;
                        }
                        for x in 0..union.len() {
                            for y in (x + 1)..union.len() {
                                for z in (y + 1)..union.len() {
                                    out.push((
                                        Constraint::Collinear {
                                            a: union[x].clone(),
                                            b: union[y].clone(),
                                            c: union[z].clone(),
                                        },
                                        vec![left_desc.clone(), right_desc.clone()],
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            Self::ParallelTransitivity => {
                for f1 in &graph.facts {
                    let Constraint::Parallel {
                        first: a,
                        second: b,
                    } = &f1.constraint
                    else {
                        continue;
                    };
                    for f2 in &graph.facts {
                        let Constraint::Parallel {
                            first: c,
                            second: d,
                        } = &f2.constraint
                        else {
                            continue;
                        };
                        let middle = if b == c {
                            Some((a, d))
                        } else if b == d {
                            Some((a, c))
                        } else if a == c {
                            Some((b, d))
                        } else if a == d {
                            Some((b, c))
                        } else {
                            None
                        };
                        if let Some((left, right)) = middle {
                            if left != right {
                                out.push((
                                    Constraint::Parallel {
                                        first: left.clone(),
                                        second: right.clone(),
                                    },
                                    vec![f1.constraint.describe(), f2.constraint.describe()],
                                ));
                            }
                        }
                    }
                }
            }
            Self::PerpendicularTransitivity => {
                for f1 in &graph.facts {
                    let Constraint::Perpendicular {
                        first: a,
                        second: b,
                    } = &f1.constraint
                    else {
                        continue;
                    };
                    for f2 in &graph.facts {
                        let Constraint::Perpendicular {
                            first: c,
                            second: d,
                        } = &f2.constraint
                        else {
                            continue;
                        };
                        let middle = if b == c {
                            Some((a, d))
                        } else if b == d {
                            Some((a, c))
                        } else if a == c {
                            Some((b, d))
                        } else if a == d {
                            Some((b, c))
                        } else {
                            None
                        };
                        if let Some((left, right)) = middle {
                            if left != right {
                                out.push((
                                    Constraint::Parallel {
                                        first: left.clone(),
                                        second: right.clone(),
                                    },
                                    vec![f1.constraint.describe(), f2.constraint.describe()],
                                ));
                            }
                        }
                    }
                }
            }
            Self::ParallelPerpendicular => {
                for f1 in &graph.facts {
                    let Constraint::Parallel {
                        first: a,
                        second: b,
                    } = &f1.constraint
                    else {
                        continue;
                    };
                    for f2 in &graph.facts {
                        let Constraint::Perpendicular {
                            first: c,
                            second: d,
                        } = &f2.constraint
                        else {
                            continue;
                        };
                        let pairs = [(a, c, d), (b, c, d), (c, a, b), (d, a, b)];
                        for (target, first, second) in pairs {
                            let partner = if first == target {
                                second
                            } else if second == target {
                                first
                            } else {
                                continue;
                            };
                            out.push((
                                Constraint::Perpendicular {
                                    first: target.clone(),
                                    second: partner.clone(),
                                },
                                vec![f1.constraint.describe(), f2.constraint.describe()],
                            ));
                        }
                    }
                }
            }
            Self::ParallelogramDiagonals => {
                for f1 in &graph.facts {
                    let Constraint::MidpointOf {
                        p: p1,
                        a: a1,
                        b: b1,
                    } = &f1.constraint
                    else {
                        continue;
                    };
                    for f2 in &graph.facts {
                        let Constraint::MidpointOf {
                            p: p2,
                            a: a2,
                            b: b2,
                        } = &f2.constraint
                        else {
                            continue;
                        };
                        if p1 != p2 || f1 == f2 {
                            continue;
                        }
                        // the four endpoints must be distinct
                        let ends = [a1, b1, a2, b2];
                        let mut distinct = true;
                        for i in 0..4 {
                            for j in (i + 1)..4 {
                                if ends[i] == ends[j] {
                                    distinct = false;
                                }
                            }
                        }
                        if !distinct {
                            continue;
                        }
                        // the shared midpoint makes abcd a parallelogram in
                        // the order (a1, a2, b1, b2): the opposite sides are
                        // a1->a2 and b2->b1
                        let inputs = vec![f1.constraint.describe(), f2.constraint.describe()];
                        out.push((
                            Constraint::Parallel {
                                first: Segment {
                                    from: a1.clone(),
                                    to: a2.clone(),
                                },
                                second: Segment {
                                    from: b2.clone(),
                                    to: b1.clone(),
                                },
                            },
                            inputs.clone(),
                        ));
                        out.push((
                            Constraint::EqualLength {
                                first: Segment {
                                    from: a1.clone(),
                                    to: a2.clone(),
                                },
                                second: Segment {
                                    from: b2.clone(),
                                    to: b1.clone(),
                                },
                            },
                            inputs.clone(),
                        ));
                    }
                }
            }
            Self::AngleEqual {
                circle,
                center,
                a,
                b,
                c,
                d,
            } => {
                // Angles standing on the same chord of the same circle, from
                // vertices on the same side of that chord, are equal. Every one
                // of those premises is load-bearing: drop the circle and the
                // theorem is about nothing, drop the side test and the two
                // angles are supplementary rather than equal -- exactly the
                // family of false derivation the audit wanted a detector for, so
                // the rule refuses it instead of reproducing it.
                let on_circle = |p: &str| -> bool {
                    graph.has_fact(&Constraint::OnCircle {
                        p: p.to_string(),
                        circle: circle.clone(),
                    })
                };
                if !(on_circle(a) && on_circle(b) && on_circle(c) && on_circle(d)) || b == d {
                    return Ok(out);
                }
                let at_b = side(a, c, b)?;
                let at_d = side(a, c, d)?;
                if at_b == 0 || at_d == 0 || at_b != at_d {
                    return Ok(out);
                }
                let inputs = vec![
                    Constraint::OnCircle {
                        p: b.clone(),
                        circle: circle.clone(),
                    }
                    .describe(),
                    Constraint::OnCircle {
                        p: d.clone(),
                        circle: circle.clone(),
                    }
                    .describe(),
                ];
                out.push((
                    Constraint::AngleEqual {
                        first: Angle3::new(b, a, c),
                        second: Angle3::new(d, a, c),
                    },
                    inputs.clone(),
                ));
                // And the central angle over the same chord is twice the
                // inscribed one. Stated as a cosine the claim needs no case
                // split over reflex angles, because cos(2t) = cos(360 - 2t): the
                // doubled value is right whether the vertex sits on the major or
                // the minor arc. Exact, because the cosine of a lattice angle is
                // a + b sqrt(d) and so is twice it.
                //
                // `?` rather than `if let Ok(..)`: a degenerate inscribed angle
                // (a zero-length leg) has no cosine, and quietly deriving one
                // fewer fact is the worse failure here. The kernel's contract
                // is to refuse a figure it cannot speak exactly about, and the
                // sibling `angle_terms(..)?` above already propagates for the
                // same reason. Swallowing it let saturation "succeed" with the
                // theorem's conclusion missing, and nothing downstream could
                // tell that from a figure where the rule does not apply.
                let cos = graph.angle_terms(&Angle3::new(b, a, c))?.cosine()?;
                let doubled = cos
                    .mul(&cos)?
                    .scale(&Q::from_int(2))?
                    .sub(&QSqrt::rational(Frac::from_int(1)))?;
                out.push((
                    Constraint::AngleIs {
                        at: Angle3::new(center, a, c),
                        cos: doubled,
                    },
                    inputs,
                ));
            }
            Self::AngleIsosceles { apex, left, right } => {
                // Equal legs, equal opposite angles -- of a triangle. The
                // non-degeneracy premise is not ceremony: with the three points
                // collinear the two "base angles" are 0 and 180 degrees and the
                // classical statement says nothing, so the rule asks for the
                // triangle fact before it will speak.
                let frame = Constraint::Triangle {
                    a: apex.clone(),
                    b: left.clone(),
                    c: right.clone(),
                };
                let legs = Constraint::EqualLength {
                    first: Segment {
                        from: apex.clone(),
                        to: left.clone(),
                    },
                    second: Segment {
                        from: apex.clone(),
                        to: right.clone(),
                    },
                };
                if !(graph.has_fact(&frame) && graph.has_fact(&legs)) {
                    return Ok(out);
                }
                let inputs = vec![frame.describe(), legs.describe()];
                out.push((
                    Constraint::AngleEqual {
                        first: Angle3::new(left, apex, right),
                        second: Angle3::new(right, left, apex),
                    },
                    inputs,
                ));
            }
            Self::MidpointBisectsPerpendicular {
                apex,
                midpoint,
                left,
                right,
            } => {
                // The median that is also an altitude makes the triangle
                // isosceles, and then the base angles follow. All three premises
                // are required: without the triangle the apex may sit on the
                // base itself, where the "perpendicular median" has zero length
                // and the conclusion is vacuous.
                let bisector = Constraint::MidpointOf {
                    p: midpoint.clone(),
                    a: left.clone(),
                    b: right.clone(),
                };
                let square = Constraint::Perpendicular {
                    first: Segment {
                        from: apex.clone(),
                        to: midpoint.clone(),
                    },
                    second: Segment {
                        from: left.clone(),
                        to: right.clone(),
                    },
                };
                let frame = Constraint::Triangle {
                    a: apex.clone(),
                    b: left.clone(),
                    c: right.clone(),
                };
                if !(graph.has_fact(&bisector) && graph.has_fact(&square) && graph.has_fact(&frame))
                {
                    return Ok(out);
                }
                let inputs = vec![bisector.describe(), square.describe(), frame.describe()];
                out.push((
                    Constraint::EqualLength {
                        first: Segment {
                            from: apex.clone(),
                            to: left.clone(),
                        },
                        second: Segment {
                            from: apex.clone(),
                            to: right.clone(),
                        },
                    },
                    inputs.clone(),
                ));
                out.push((
                    Constraint::AngleEqual {
                        first: Angle3::new(left, apex, right),
                        second: Angle3::new(right, left, apex),
                    },
                    inputs,
                ));
            }
            Self::EqualAnglesSubtendArc {
                circle,
                chord_left,
                chord_right,
                first,
                second,
            } => {
                // The converse of the chord rule: one vertex already sits on the
                // circle through the chord, the other sees that chord under the
                // same angle from the same side, so the other sits on it too. The
                // side test is what makes the converse true -- equal angles from
                // opposite sides describe a mirror image rather than the same
                // arc -- and the conclusion is a fact the predicate checks by
                // distance, not a claim about a circle nobody constructed.
                let seen = Constraint::AngleEqual {
                    first: Angle3::new(first, chord_left, chord_right),
                    second: Angle3::new(second, chord_left, chord_right),
                };
                let anchor = Constraint::OnCircle {
                    p: first.clone(),
                    circle: circle.clone(),
                };
                if !(graph.has_fact(&seen) && graph.has_fact(&anchor)) || first == second {
                    return Ok(out);
                }
                let one = side(chord_left, chord_right, first)?;
                let other = side(chord_left, chord_right, second)?;
                if one == 0 || other == 0 || one != other {
                    return Ok(out);
                }
                out.push((
                    Constraint::OnCircle {
                        p: second.clone(),
                        circle: circle.clone(),
                    },
                    vec![seen.describe(), anchor.describe()],
                ));
            }
        }
        Ok(out)
    }

    /// Every concrete instance of this rule that the scene offers, obtained by
    /// filling blank names from the facts: an empty field is a wildcard.
    ///
    /// Saturation needs this spelled out. A rule set compiled into a binary is a
    /// table of all-blank rules -- `RULES` below is exactly that -- and an
    /// all-blank rule matches no fact, fires nothing, and reports nothing. A rule
    /// table that quietly never fires is the audit's central complaint wearing a
    /// new costume: reasoning whose failure looks like success. So the expansion
    /// happens here, where it can be counted and tested, and it is driven by the
    /// facts rather than by the points: the chord rule only ever binds vertices
    /// some fact puts on the circle, which keeps the search linear in what the
    /// figure actually states instead of cubic in how many names it happens to
    /// contain.
    pub fn instantiate(&self, graph: &SceneGraph) -> Vec<Rule> {
        match self {
            Self::AngleEqual {
                circle,
                center,
                a,
                b,
                c,
                d,
            } => {
                if !(circle.is_empty()
                    || center.is_empty()
                    || a.is_empty()
                    || b.is_empty()
                    || c.is_empty()
                    || d.is_empty())
                {
                    return vec![self.clone()];
                }
                let mut out = Vec::new();
                for k in graph
                    .circles
                    .iter()
                    .filter(|k| circle.is_empty() || &k.name == circle)
                {
                    let middle = if center.is_empty() {
                        k.center.clone()
                    } else {
                        center.clone()
                    };
                    let on: Vec<&String> = graph
                        .facts
                        .iter()
                        .filter_map(|f| match &f.constraint {
                            Constraint::OnCircle { p, circle: name } if name == &k.name => Some(p),
                            _ => None,
                        })
                        .collect();
                    for chord_first in &on {
                        for chord_second in &on {
                            if chord_first == chord_second {
                                continue;
                            }
                            for left in &on {
                                if left == chord_first || left == chord_second {
                                    continue;
                                }
                                for right in &on {
                                    if right == chord_first
                                        || right == chord_second
                                        || right == left
                                    {
                                        continue;
                                    }
                                    out.push(Self::AngleEqual {
                                        circle: k.name.clone(),
                                        center: middle.clone(),
                                        a: (*chord_first).clone(),
                                        b: (*left).clone(),
                                        c: (*chord_second).clone(),
                                        d: (*right).clone(),
                                    });
                                }
                            }
                        }
                    }
                }
                out
            }
            _ => self.instantiate_other(graph),
        }
    }

    fn instantiate_other(&self, graph: &SceneGraph) -> Vec<Rule> {
        match self {
            Self::AngleIsosceles { apex, left, right } => {
                if !(apex.is_empty() || left.is_empty() || right.is_empty()) {
                    return vec![self.clone()];
                }
                let mut out = Vec::new();
                for fact in &graph.facts {
                    let Constraint::EqualLength { first, second } = &fact.constraint else {
                        continue;
                    };
                    // the apex is wherever the two equal segments meet
                    let Some((peak, one, other)) = shared_endpoint(first, second) else {
                        continue;
                    };
                    out.push(Self::AngleIsosceles {
                        apex: peak,
                        left: one,
                        right: other,
                    });
                }
                out
            }
            Self::MidpointBisectsPerpendicular {
                apex,
                midpoint,
                left,
                right,
            } => {
                if !(apex.is_empty() || midpoint.is_empty() || left.is_empty() || right.is_empty())
                {
                    return vec![self.clone()];
                }
                let mut out = Vec::new();
                for fact in &graph.facts {
                    let Constraint::MidpointOf { p, a, b } = &fact.constraint else {
                        continue;
                    };
                    for candidate in &graph.points {
                        if candidate.name == *p || candidate.name == *a || candidate.name == *b {
                            continue;
                        }
                        out.push(Self::MidpointBisectsPerpendicular {
                            apex: candidate.name.clone(),
                            midpoint: p.clone(),
                            left: a.clone(),
                            right: b.clone(),
                        });
                    }
                }
                out
            }
            Self::EqualAnglesSubtendArc {
                circle,
                chord_left,
                chord_right,
                first,
                second,
            } => {
                if !(circle.is_empty()
                    || chord_left.is_empty()
                    || chord_right.is_empty()
                    || first.is_empty()
                    || second.is_empty())
                {
                    return vec![self.clone()];
                }
                let mut out = Vec::new();
                for fact in &graph.facts {
                    // the premise is an equality of two angles standing on one
                    // chord; a pair of unrelated angles is not an instance
                    let Constraint::AngleEqual {
                        first: one,
                        second: other,
                    } = &fact.constraint
                    else {
                        continue;
                    };
                    // The chord is shared when the two angles stand on the same unordered pair of
                    // endpoints. Either orientation qualifies; the endpoints are
                    // recorded in the first angle's order because an arc has no
                    // preferred direction.
                    let shares_endpoints = (one.from == other.from && one.to == other.to)
                        || (one.from == other.to && one.to == other.from);
                    if !shares_endpoints {
                        continue;
                    }
                    let (left, right) = (one.from.clone(), one.to.clone());
                    for k in graph
                        .circles
                        .iter()
                        .filter(|k| circle.is_empty() || &k.name == circle)
                    {
                        out.push(Self::EqualAnglesSubtendArc {
                            circle: k.name.clone(),
                            chord_left: left.clone(),
                            chord_right: right.clone(),
                            first: one.at.clone(),
                            second: other.at.clone(),
                        });
                    }
                }
                out
            }
            _ => vec![self.clone()],
        }
    }

    /// Fire every rule until the fact set stops growing (the symbolic
    /// engine's saturation loop). Returns the facts added on top of the
    /// givens, each with its rule certificate. Rules are first expanded into
    /// the concrete instances the scene supports; an all-blank rule is a
    /// template, never a fact-generating event, because a rule table that
    /// quietly matches nothing is how reasoning comes to look like success.
    pub fn saturate(graph: &mut SceneGraph, max_rounds: usize) -> anyhow::Result<usize> {
        let mut added = 0usize;
        for _ in 0..max_rounds {
            let mut fresh = Vec::new();
            for rule in RULES {
                for instance in rule.instantiate(graph) {
                    for (constraint, inputs) in instance.forward(graph)? {
                        if !graph.has_fact(&constraint) {
                            fresh.push((
                                constraint,
                                Provenance::Derived {
                                    rule: instance.name().to_string(),
                                    inputs,
                                },
                            ));
                        }
                    }
                }
            }
            if fresh.is_empty() {
                break;
            }
            added += fresh.len();
            for (constraint, provenance) in fresh {
                graph.add_fact(constraint, provenance);
            }
        }
        Ok(added)
    }
}

/// The shared endpoint of two segments, with the two far ends: the apex of the
/// isosceles triangle the pair forms when they are equal and not collinear.
///
/// The isosceles rule used to take its apex from the caller's good intentions,
/// and a caller that picked the wrong end of an equal pair got a "proved"
/// statement about base angles at the apex and at a base -- the same shape of
/// error as the audit's false theorem that an isosceles triangle has three equal
/// angles, arrived at by applying a true rule to the wrong vertex. Locating the
/// apex from the two segments instead of trusting a name removes the choice.
/// `None` when the segments are disjoint (two separate isosceles triangles) or
/// identical (no angle at all).
pub fn shared_endpoint(first: &Segment, second: &Segment) -> Option<(String, String, String)> {
    if first.from == second.from {
        Some((first.from.clone(), first.to.clone(), second.to.clone()))
    } else if first.from == second.to {
        Some((first.from.clone(), first.to.clone(), second.from.clone()))
    } else if first.to == second.from {
        Some((first.to.clone(), first.from.clone(), second.to.clone()))
    } else if first.to == second.to {
        Some((first.to.clone(), first.from.clone(), second.from.clone()))
    } else {
        None
    }
}

/// The verdict of randomized falsification: a counterexample rejects the
/// claim; surviving draws are evidence, never proof.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// The claim is refuted: a configuration satisfying every premise
    /// violates the claim; the counterexample's coordinates come with it.
    Counterexample {
        trial: usize,
        points: Vec<(String, Q, Q)>,
    },
    /// Every premise-satisfying draw also satisfied the claim. Evidence, not
    /// proof -- the symbolic engine still has to close it.
    Unrefuted { trials: usize, satisfying: usize },
}

/// Randomized falsification: sample uniform grid configurations, keep the
/// ones satisfying every premise exactly, and look for one violating the
/// claim (fix 9 of the geometry-reasoning audit). One counterexample rejects;
/// `Unrefuted` means the claim survived every draw -- never that it is true, and
/// `satisfying == 0` in it means the search checked *nothing* (a premises set
/// no draw can satisfy; see `falsify_with_circles`).
pub fn falsify<R: rand::Rng>(
    rng: &mut R,
    premises: &[Constraint],
    claim: &Constraint,
    trials: usize,
    grid: i64,
) -> anyhow::Result<Verdict> {
    falsify_with_circles(rng, premises, &[], claim, trials, grid)
}

/// Falsification for figures whose premises sit on named circles.
///
/// A circle premise cannot be *hoped* for any more than a midpoint can: a
/// uniform integer draw lands on a stated circle essentially never, so a
/// falsifier that free-sampled `OnCircle` points would report "survived 4096
/// draws" about a search that never once satisfied the premise -- the vacuous
/// pass the audit was written about. Such a point is instead constructed onto
/// the circle exactly: with rational `t`, `((1-t^2)/(1+t^2), 2t/(1+t^2))` is a
/// rational point of the unit circle, so centre plus radius times it is an exact
/// rational point of the circle being reasoned about. That needs a *rational*
/// radius, i.e. a radius squared that is the square of a rational -- a
/// requirement on the figure, said out loud instead of surfacing as a silently
/// empty search.
pub fn falsify_with_circles<R: rand::Rng>(
    rng: &mut R,
    premises: &[Constraint],
    circles: &[KCircle],
    claim: &Constraint,
    trials: usize,
    grid: i64,
) -> anyhow::Result<Verdict> {
    anyhow::ensure!(trials > 0 && grid > 0, "trials and grid must be positive");
    // A figure may declare its circle as a premise (`Circle`) or carry one the
    // caller's scene already named; the sampler needs centre and radius from
    // either source, so both go into one pool before anything asks for a name.
    let mut pool: Vec<KCircle> = circles.to_vec();
    for premise in premises.iter().chain(std::iter::once(claim)) {
        if let Constraint::Circle {
            name,
            center,
            radius_sq,
        } = premise
        {
            if !pool.iter().any(|c| c.name == *name) {
                pool.push(KCircle {
                    name: name.clone(),
                    center: center.clone(),
                    radius_sq: *radius_sq,
                });
            }
        }
    }
    let circle_of = |name: &str| -> anyhow::Result<&KCircle> {
        pool.iter()
            .find(|c| c.name == name)
            .ok_or_else(|| GeometryError::UnknownCircle(name.to_string()).into())
    };
    // Premises the sampler can never satisfy are a misuse, not a result: an
    // angle-equality, congruence or equal-area claim holds on a set of
    // configurations of measure zero inside the free draws, so a run over them
    // returns "nothing violated it" and "nothing satisfied it" in the same
    // breath. Refuse, and say what to pass instead.
    for premise in premises {
        if unsamplable(premise) {
            anyhow::bail!(
                "the falsifier cannot sample the premise {}; sample the figure's incidence and \
                 metric givens and make the angle or area claim the conclusion",
                premise.describe()
            );
        }
        if let Constraint::OnCircle { circle, .. } = premise {
            circle_of(circle)?;
        }
    }
    if let Constraint::OnCircle { circle, .. } = claim {
        circle_of(circle)?;
    }
    // A midpoint premise *determines* its point: sample it by construction
    // (the exact rational midpoint), never by hoping a free draw lands on it.
    let mut mentioned: Vec<String> = Vec::new();
    let mut push_name = |name: &str| {
        if !mentioned.iter().any(|n| n == name) {
            mentioned.push(name.to_string());
        }
    };
    for constraint in premises.iter().chain(std::iter::once(claim)) {
        for name in constraint_point_names(constraint) {
            push_name(&name);
        }
        // The centre of a circle in play is a point of the figure too: it has to
        // be placed for the circle to exist anywhere.
        for constraint in std::iter::once(constraint) {
            if let Constraint::OnCircle { circle, .. } = constraint {
                push_name(&circle_of(circle)?.center);
            }
        }
    }
    let mut constructed: Vec<(String, String, String)> = Vec::new();
    let mut anchored: Vec<(String, String, Q)> = Vec::new();
    for premise in premises {
        match premise {
            Constraint::MidpointOf { p, a, b } => {
                constructed.push((p.clone(), a.clone(), b.clone()));
            }
            Constraint::OnCircle { p, circle } => {
                let found = circle_of(circle)?;
                anyhow::ensure!(
                    !constructed.iter().any(|(m, _, _)| m == p),
                    "the point {} is determined by both a midpoint and a circle premise; the \
                     sampler cannot satisfy both by construction",
                    p
                );
                let radius = rational_sqrt(&found.radius_sq).ok_or_else(|| {
                    anyhow::anyhow!(
                        "circle {} has radius squared {}, which is not the square of a rational; \
                         the falsifier can only place points exactly on a circle of rational \
                         radius, so restate the figure with one",
                        found.name,
                        found.radius_sq
                    )
                })?;
                if radius.is_zero() {
                    return Err(GeometryError::ZeroRadiusCircle {
                        circle: found.name.clone(),
                    }
                    .into());
                }
                anchored.push((p.clone(), found.center.clone(), radius));
            }
            _ => {}
        }
    }
    let drawn: Vec<String> = mentioned
        .iter()
        .filter(|name| {
            !constructed.iter().any(|(p, _, _)| p == *name)
                && !anchored.iter().any(|(p, _, _)| p == *name)
        })
        .cloned()
        .collect();
    let mut satisfying = 0usize;
    for trial in 0..trials {
        // small integer coordinates keep the sampled configurations honest
        // and repeatable without any floating point
        let mut coords: Vec<(String, Q, Q)> = Vec::with_capacity(mentioned.len());
        for name in &drawn {
            coords.push((
                name.clone(),
                Q::from_int(rng.random_range(0..=grid)),
                Q::from_int(rng.random_range(0..=grid)),
            ));
        }
        for (p, a, b) in &constructed {
            let found = coords_of(&coords, a)?;
            let other = coords_of(&coords, b)?;
            coords.push((
                p.clone(),
                found.0.add(&other.0)?.half()?,
                found.1.add(&other.1)?.half()?,
            ));
        }
        // Points a circle premise determines are put *on* the circle exactly, via
        // the rational parametrisation, rather than hoped for. The parameter is
        // the only random choice here, and every value of it produces a point
        // whose distance from the centre is the radius -- by identity, not by
        // luck.
        for (point, center, radius) in &anchored {
            let middle = coords_of(&coords, center).map_err(|_| {
                anyhow::anyhow!(
                    "the centre {} of the circle carrying {} is itself determined by a circle \
                     premise; the sampler cannot place a chain of such points",
                    center,
                    point
                )
            })?;
            let t = Q::from_int(rng.random_range(0..=grid));
            let one = Q::from_int(1);
            let square = t.mul(&t)?;
            let denominator = one.add(&square)?;
            let cosine = one.sub(&square)?.div(&denominator)?;
            let sine = t.mul(&one.add(&one)?)?.div(&denominator)?;
            coords.push((
                point.clone(),
                middle.0.add(&radius.mul(&cosine)?)?,
                middle.1.add(&radius.mul(&sine)?)?,
            ));
        }
        // distinct points: reject the draw otherwise
        let mut ok = true;
        for i in 0..coords.len() {
            for j in (i + 1)..coords.len() {
                if coords[i].1 == coords[j].1 && coords[i].2 == coords[j].2 {
                    ok = false;
                }
            }
        }
        if !ok {
            continue;
        }
        let mut premises_hold = true;
        for premise in premises {
            if !constraint_holds(&coords, premise)? {
                premises_hold = false;
                break;
            }
        }
        if !premises_hold {
            continue;
        }
        satisfying += 1;
        if !constraint_holds(&coords, claim)? {
            return Ok(Verdict::Counterexample {
                trial,
                points: coords,
            });
        }
    }
    Ok(Verdict::Unrefuted { trials, satisfying })
}

fn coords_of(coords: &[(String, Q, Q)], name: &str) -> anyhow::Result<(Q, Q)> {
    coords
        .iter()
        .find(|(n, _, _)| n == name)
        .map(|(_, x, y)| (*x, *y))
        .ok_or_else(|| anyhow::anyhow!("unconstructed point '{name}'"))
}

/// Whether a constraint is exactly *decidable* but not exactly *sampleable*.
/// Angle, congruence and area statements hold on a measure-zero slice of the
/// grid, so a falsifier that took them as premises would draw a thousand
/// configurations, satisfy none, and report a clean bill of health. The kernel
/// says which statements belong on which side of that line rather than leaving
/// the caller to read an empty `satisfying` count off a verdict.
fn unsamplable(constraint: &Constraint) -> bool {
    matches!(
        constraint,
        Constraint::AngleEqual { .. }
            | Constraint::AngleIs { .. }
            | Constraint::AreaEqual { .. }
            | Constraint::Congruent { .. }
    )
}

/// The exact square root of a rational when one exists: integer square roots of
/// numerator and denominator taken separately. A floating estimate may suggest
/// where to look, but the candidate is *proved* by integer multiplication before
/// it is returned, so the answer is exact or absent -- never a float standing in
/// for a radius (audit fix 7).
pub fn rational_sqrt(value: &Frac) -> Option<Q> {
    fn root_of(n: i128) -> Option<i128> {
        if n < 0 {
            return None;
        }
        if n == 0 {
            return Some(0);
        }
        let guess = (n as f64).sqrt() as i128;
        let mut candidate = guess.saturating_sub(2).max(0);
        while candidate <= guess + 2 {
            if let Some(square) = candidate.checked_mul(candidate) {
                if square == n {
                    return Some(candidate);
                }
            }
            candidate += 1;
        }
        None
    }
    Q::new(root_of(value.num)?, root_of(value.den)?).ok()
}

/// Every point name a constraint mentions -- the falsifier's and the sampler's
/// view of what has to be placed. Kept in the kernel so that a sampler in this
/// module and a sampler in `verify` or the CLI cannot disagree about which names
/// a statement contains.
pub fn constraint_point_names(constraint: &Constraint) -> Vec<String> {
    let refs: Vec<&String> = match constraint {
        Constraint::Collinear { a, b, c }
        | Constraint::NonCollinear { a, b, c }
        | Constraint::Triangle { a, b, c } => vec![a, b, c],
        Constraint::Between { a, m, b } | Constraint::MidpointOf { p: m, a, b } => {
            vec![a, m, b]
        }
        Constraint::RatioOf { p, a, b, .. } => vec![p, a, b],
        Constraint::Distinct { a, b } => vec![a, b],
        Constraint::Parallel { first, second }
        | Constraint::Perpendicular { first, second }
        | Constraint::EqualLength { first, second }
        | Constraint::ScaleLength { first, second, .. } => {
            vec![&first.from, &first.to, &second.from, &second.to]
        }
        Constraint::LengthIs { seg, .. } => vec![&seg.from, &seg.to],
        Constraint::AngleEqual { first, second } => vec![
            &first.at,
            &first.from,
            &first.to,
            &second.at,
            &second.from,
            &second.to,
        ],
        Constraint::RightAngle { at } | Constraint::AngleIs { at, .. } => {
            vec![&at.at, &at.from, &at.to]
        }
        Constraint::AreaEqual { first, second } | Constraint::Congruent { first, second } => {
            vec![
                &first.a, &first.b, &first.c, &second.a, &second.b, &second.c,
            ]
        }
        Constraint::Circle { center, .. } => vec![center],
        Constraint::OnCircle { p, .. } => vec![p],
        Constraint::Diameter { a, b, .. } => vec![a, b],
    };
    let mut names: Vec<String> = refs.into_iter().cloned().collect();
    names.sort();
    names.dedup();
    names
}

/// Whether a constraint mentions a point by name. Samplers in `verify`, the CLI
/// and the vision layer all need this, and two implementations that disagree
/// about which names a statement contains is how a checker ends up placing a
/// point its predicate then reads as missing.
pub fn constraint_mentions(constraint: &Constraint, name: &str) -> bool {
    constraint_point_names(constraint).iter().any(|n| n == name)
}

/// Build a scene graph from raw coordinates and a list of constraints marked
/// `given` -- the counterexample sampler's inverse, and the shape the CLI
/// accepts on disk.
pub fn graph_from_points(points: Vec<KPoint>, constraints: Vec<Constraint>) -> SceneGraph {
    SceneGraph {
        points,
        facts: constraints.into_iter().map(Fact::given).collect(),
        circles: Vec::new(),
        geometry: euclidean_by_default(),
        not_established: Vec::new(),
    }
}

/// Free-function wrapper over [`Rule::saturate`] so callers can saturate a
/// graph without naming the rule enum.
///
/// Returns the number of derived facts, or the error the saturating loop
/// raised. It does not swallow it: a rule that cannot be evaluated is a scene
/// the caller is not entitled to trust, and a silent `0` here would report a
/// graph as saturated when nothing was derived at all.
pub fn saturate_rules(graph: &mut SceneGraph, max_rounds: usize) -> anyhow::Result<usize> {
    Rule::saturate(graph, max_rounds)
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

    fn q(num: i128, den: i128) -> Q {
        Q::new(num, den).unwrap()
    }

    #[test]
    fn test_rationals_are_exact_where_floats_are_not() {
        // 1/10 + 2/10 = 3/10 exactly; in f32 the same expression is off
        assert_eq!(q(1, 10).add(&q(2, 10)).unwrap(), q(3, 10));
        assert_ne!((0.1f64) + (0.2f64), 0.3f64);
        // normalization: 2/4 == 1/2, and the negation moves the sign up
        assert_eq!(q(2, 4), q(1, 2));
        assert_eq!(q(1, -2), q(-1, 2));
        // ordering is exact
        assert!(q(1, 3).less(&q(1, 2)));
        assert!(!q(2, 6).less(&q(1, 3)));
        // zero denominators refuse
        assert!(Q::new(1, 0).is_err());
    }

    #[test]
    fn test_intersection_is_exact() {
        // line y = x + 1/3 through (0, 1/3) and (1, 4/3); line x = 2/3
        // through (2/3, 0) and (2/3, 1): the intersection is exactly
        // (2/3, 1) -- a float kernel would return 0.9999999...
        let points = vec![
            KPoint {
                name: "A".into(),
                x: Frac { num: 0, den: 1 },
                y: Frac { num: 1, den: 3 },
            },
            KPoint {
                name: "B".into(),
                x: Frac { num: 1, den: 1 },
                y: Frac { num: 4, den: 3 },
            },
            KPoint {
                name: "C".into(),
                x: Frac { num: 2, den: 3 },
                y: Frac { num: 0, den: 1 },
            },
            KPoint {
                name: "D".into(),
                x: Frac { num: 2, den: 3 },
                y: Frac { num: 1, den: 1 },
            },
        ];
        let graph = graph_from_points(points, vec![]);
        let point = construct(
            &graph,
            &Construction::Intersection {
                first: Segment {
                    from: "A".into(),
                    to: "B".into(),
                },
                second: Segment {
                    from: "C".into(),
                    to: "D".into(),
                },
            },
            "E",
        )
        .unwrap();
        assert_eq!(point.x.num, 2);
        assert_eq!(point.x.den, 3);
        assert_eq!(point.y.num, 1);
        assert_eq!(point.y.den, 1);
    }

    #[test]
    fn test_intersect_refuses_parallel_and_coincident() {
        let points = vec![
            KPoint {
                name: "A".into(),
                x: Frac::from_int(0),
                y: Frac::from_int(0),
            },
            KPoint {
                name: "B".into(),
                x: Frac::from_int(1),
                y: Frac::from_int(1),
            },
            KPoint {
                name: "C".into(),
                x: Frac::from_int(0),
                y: Frac::from_int(1),
            },
            KPoint {
                name: "D".into(),
                x: Frac::from_int(1),
                y: Frac::from_int(2),
            },
            KPoint {
                name: "E".into(),
                x: Frac::from_int(2),
                y: Frac::from_int(2),
            },
        ];
        let graph = graph_from_points(points, vec![]);
        let parallel = construct(
            &graph,
            &Construction::Intersection {
                first: Segment {
                    from: "A".into(),
                    to: "B".into(),
                },
                second: Segment {
                    from: "C".into(),
                    to: "D".into(),
                },
            },
            "F",
        );
        assert!(matches!(
            parallel.unwrap_err().downcast_ref::<GeometryError>(),
            Some(GeometryError::ParallelLines)
        ));
        let coincident = construct(
            &graph,
            &Construction::Intersection {
                first: Segment {
                    from: "A".into(),
                    to: "B".into(),
                },
                second: Segment {
                    from: "A".into(),
                    to: "E".into(),
                },
            },
            "F",
        );
        assert!(matches!(
            coincident.unwrap_err().downcast_ref::<GeometryError>(),
            Some(GeometryError::CoincidentLines)
        ));
    }

    #[test]
    fn test_rules_fire_with_certificates_and_stop() {
        // D is the midpoint of BC: the kernel derives the collinearity and
        // the length equality, each naming the rule and its input
        let points = vec![
            KPoint {
                name: "B".into(),
                x: Frac::from_int(0),
                y: Frac::from_int(0),
            },
            KPoint {
                name: "C".into(),
                x: Frac::from_int(4),
                y: Frac::from_int(2),
            },
            KPoint {
                name: "D".into(),
                x: Frac::from_int(2),
                y: Frac::from_int(1),
            },
        ];
        let mut graph = graph_from_points(
            points,
            vec![Constraint::MidpointOf {
                p: "D".into(),
                a: "B".into(),
                b: "C".into(),
            }],
        );
        let added = Rule::saturate(&mut graph, 8).unwrap();
        assert_eq!(added, 2);
        assert!(graph.has_fact(&Constraint::Collinear {
            a: "B".into(),
            b: "D".into(),
            c: "C".into()
        }));
        assert!(graph.has_fact(&Constraint::EqualLength {
            first: Segment {
                from: "B".into(),
                to: "D".into()
            },
            second: Segment {
                from: "D".into(),
                to: "C".into()
            },
        }));
        // saturation is a fixed point: a second run adds nothing
        assert_eq!(Rule::saturate(&mut graph, 8).unwrap(), 0);
        // and every derived fact carries its certificate
        let derived = graph
            .facts
            .iter()
            .filter(|f| matches!(f.provenance, Provenance::Derived { .. }))
            .count();
        assert_eq!(derived, 2);
    }

    #[test]
    fn test_parallelogram_rule_and_exact_constructions() {
        // a parallelogram abcd: a=(0,0), b=(4,0), d=(1,2), c=b+d=(5,2); the
        // diagonals ac and bd share the midpoint (5/2, 1)
        let points = vec![
            KPoint {
                name: "A".into(),
                x: Frac::from_int(0),
                y: Frac::from_int(0),
            },
            KPoint {
                name: "B".into(),
                x: Frac::from_int(4),
                y: Frac::from_int(0),
            },
            KPoint {
                name: "C".into(),
                x: Frac::from_int(5),
                y: Frac::from_int(2),
            },
            KPoint {
                name: "D".into(),
                x: Frac::from_int(1),
                y: Frac::from_int(2),
            },
            KPoint {
                name: "P".into(),
                x: Frac { num: 5, den: 2 },
                y: Frac { num: 1, den: 1 },
            },
        ];
        let mut graph = graph_from_points(
            points,
            vec![
                Constraint::MidpointOf {
                    p: "P".into(),
                    a: "A".into(),
                    b: "C".into(),
                },
                Constraint::MidpointOf {
                    p: "P".into(),
                    a: "B".into(),
                    b: "D".into(),
                },
            ],
        );
        Rule::saturate(&mut graph, 8).unwrap();
        let ab = Segment {
            from: "A".into(),
            to: "B".into(),
        };
        let cd = Segment {
            from: "D".into(),
            to: "C".into(),
        };
        assert!(graph.has_fact(&Constraint::Parallel {
            first: ab.clone(),
            second: cd.clone()
        }));
        assert!(graph.has_fact(&Constraint::EqualLength {
            first: ab,
            second: cd
        }));
        // the constructed midpoint of AC is exactly P
        let mid = construct(
            &graph,
            &Construction::Midpoint {
                a: "A".into(),
                b: "C".into(),
            },
            "M",
        )
        .unwrap();
        assert_eq!((mid.x.num, mid.x.den, mid.y.num, mid.y.den), (5, 2, 1, 1));
    }

    #[test]
    fn test_falsify_rejects_a_false_claim() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        // claim: len(AB) = len(AC), with no premises forcing it
        let claim = Constraint::EqualLength {
            first: Segment {
                from: "A".into(),
                to: "B".into(),
            },
            second: Segment {
                from: "A".into(),
                to: "C".into(),
            },
        };
        let mut rng = StdRng::seed_from_u64(4);
        let verdict = falsify(&mut rng, &[], &claim, 256, 6).unwrap();
        assert!(
            matches!(verdict, Verdict::Counterexample { .. }),
            "an unconstrained length claim must be falsified"
        );
        // claim: collinear(A, B, C) when A is the midpoint of BC -- every
        // premise-satisfying draw satisfies it
        let premises = vec![Constraint::MidpointOf {
            p: "A".into(),
            a: "B".into(),
            b: "C".into(),
        }];
        let claim = Constraint::Collinear {
            a: "B".into(),
            b: "A".into(),
            c: "C".into(),
        };
        let verdict = falsify(&mut rng, &premises, &claim, 128, 6).unwrap();
        match verdict {
            Verdict::Unrefuted { satisfying, trials } => {
                assert!(
                    satisfying > 0,
                    "the sampler found no valid configurations of {trials}"
                );
            }
            other => panic!("the midpoint claim is a theorem; falsified anyway: {other:?}"),
        }
    }

    #[test]
    fn test_projection_and_reflection_are_exact() {
        // project (1, 3) onto the x-axis (through (0,0),(4,0)): (1, 0);
        // reflect: (1, -3) -- exact integers, no drift
        let points = vec![
            KPoint {
                name: "P".into(),
                x: Frac::from_int(1),
                y: Frac::from_int(3),
            },
            KPoint {
                name: "A".into(),
                x: Frac::from_int(0),
                y: Frac::from_int(0),
            },
            KPoint {
                name: "B".into(),
                x: Frac::from_int(4),
                y: Frac::from_int(0),
            },
        ];
        let graph = graph_from_points(points, vec![]);
        let foot = construct(
            &graph,
            &Construction::Projection {
                p: "P".into(),
                a: "A".into(),
                b: "B".into(),
            },
            "F",
        )
        .unwrap();
        assert_eq!((foot.x.num, foot.y.num), (1, 0));
        let mirror = construct(
            &graph,
            &Construction::Reflect {
                p: "P".into(),
                a: "A".into(),
                b: "B".into(),
            },
            "R",
        )
        .unwrap();
        assert_eq!((mirror.x.num, mirror.y.num), (1, -3));
    }
}
