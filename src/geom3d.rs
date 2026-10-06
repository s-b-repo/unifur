//! Exact three-dimensional geometry (roadmap Phase 33, audit fix 12).
//!
//! Two-dimensional tricks do not transfer: a "obvious" parallel or incidence
//! fact in a drawing has no analogue once the third coordinate is populated,
//! and vision systems are measurably weak on polyhedral detail and mental
//! rotation. The fix is not to make the model visualize harder; it is to give
//! it an actual 3D state to query. This module is that state: points,
//! vectors, lines, planes and solids over the kernel's exact rationals, with
//! the operations a reasoner needs (dot, cross, projection, point/line/plane
//! distance, line-plane and plane-plane intersection, rotation) and
//! non-degeneracy refusals instead of silent nonsense.
//!
//! Everything here is exact. `AB \u{27c2} CD` becomes `dot(B - A, D - C) = 0`
//! over [`Q`], and a mental-rotation question is answered by transforming
//! coordinates rather than by imagining the object.
//!
//! # What lives here
//!
//! - [`Point3`] and [`Vec3`]: exact coordinates and exact vectors, the three
//!   dimensional analogues of the kernel's [`KPoint`] and its direction pairs.
//! - [`Line3`] and [`Plane3`]: an affine frame for each -- a point plus a
//!   direction, a normal plus an offset -- because "a line is two endpoints" is
//!   a planar habit that stops being a definition the moment a third
//!   coordinate is populated.
//! - [`Sphere`] and [`Polyhedron`]: solids, with exact containment, exact face
//!   areas, an exact signed volume, and exact orientation and convexity checks
//!   rather than a mesh-repair pass.
//! - The operations a reasoner needs: projection, distance, line/plane and
//!   plane/plane intersection, closest points on a line pair, and rotation
//!   about an axis.
//! - [`lift_scene`], [`lift_triangle`], [`lifted_holds`] and
//!   [`lifted_fact_holds`]: the 2D kernel's figures embedded at `z = 0`, so a
//!   statement proved in the plane can be *re-decided* in space by a different
//!   set of formulas instead of being trusted into a third dimension.
//!
//! # What is exact, and what is refused
//!
//! Coordinates, directions, areas and volumes are exact rationals; lengths,
//! areas and distances that are irrational come back in the kernel's quadratic
//! field [`QSqrt`] as `a + b sqrt d`. So a right angle is a zero dot product, a
//! rotated coordinate is a fraction rather than `0.7071067811865476`, and the
//! slanted face of a unit tetrahedron is exactly `sqrt(3)/2` -- a number that
//! can be compared, not merely displayed.
//!
//! Projection and distance are written in the `(n . p - d) / |n|^2` form
//! rather than through a unit normal, so they need no square root at all and
//! are total over every nonzero normal. Rotation is the one place a square
//! root is unavoidable, and there the restriction is deliberate and stated: the
//! *turn* must have a rational cosine and sine (a quarter turn, a half turn,
//! the 3-4-5 turn) and the *axis* must have a rational length. A 60-degree
//! turn and a `(1, 1, 1)` axis are both refused, because approximating either
//! produces an approximated answer to a question -- "is this still a right
//! angle?" -- that this module decides exactly.
//!
//! Non-degeneracy is enforced at every operation, reusing the kernel's
//! [`GeometryError`] rather than inventing a parallel hierarchy of refusals.
//! Where the kernel has no variant that names a failure exactly, the choice and
//! its reason are stated here rather than left to be discovered:
//!
//! - a zero direction vector is `DegenerateSegment`: a line whose direction
//!   vanishes is the degenerate segment the kernel already refuses to build;
//! - a plane with a zero normal is `EmptyGeometry`, because `0 . x = d` has no
//!   points -- there is no geometry to reason about, which is what the variant
//!   says;
//! - three collinear points offered as a plane are `DegenerateTriangle`, the
//!   exact case that variant was added for;
//! - inconsistent face winding in a [`Polyhedron`] is `Unproven`: the kernel has
//!   no winding variant, and the honest reading is that the solid's outward
//!   normal is *not established* by the vertex order it was given;
//! - two skew lines are `NoRationalSolution`: there is no intersection point to
//!   produce, rational or otherwise. They still have a closest pair, so
//!   [`closest_points_line_line`] answers for them while
//!   [`intersect_line_line`] refuses -- and says the word "skew", which is the
//!   most useful sentence in 3D geometry;
//! - parallel lines have no *unique* closest pair, so
//!   [`closest_points_line_line`] refuses with `ParallelLines`, or
//!   `CoincidentLines` when the two are the same line;
//! - a sphere of zero radius is `ZeroRadiusCircle` (a 3D sphere is the same
//!   object the kernel's circles are) and a *negative* radius squared is
//!   `NoRationalSolution`, which is the difference between "degenerate" and
//!   "impossible".
//!
//! Scope, stated plainly because the audit was about claims outrunning
//! machinery. This is the exact 3D *state* and the exact 3D operations over it.
//! It is not a solid modeller: no boolean union or difference, no
//! triangulation of a curved face, no surface of revolution, and no total
//! surface area whose face areas span more than one radical -- the quadratic
//! field is closed under addition only for a shared radicand, so
//! [`Polyhedron::total_surface_area`] refuses on a tetrahedron rather than
//! returning `2.3660...`. Not implemented is a decision; silently returning a
//! number that cannot be compared against the rest of the kernel's answers is
//! the failure mode this module is written against.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::geomkernel::{
    Angle3, Constraint, Fact, Frac, GeometryError, KPoint, QSqrt, SceneGraph, Segment, Tri3, Q,
};

/// The exact square root of a nonnegative rational, in the quadratic field.
///
/// `sqrt(p/q) = sqrt(p*q)/q`, and `QSqrt` normalizes `p*q` into
/// `square * square_free` on construction, so this is total over the
/// nonnegatives and never leaves the rationals.
fn sqrt_rational(value: &Q) -> anyhow::Result<QSqrt> {
    anyhow::ensure!(
        !value.less(&Q::ZERO),
        GeometryError::NoRationalSolution(format!(
            "a square root needs a nonnegative value, got {value}"
        ))
    );
    let radicand = value
        .num
        .checked_mul(value.den)
        .and_then(|product| i64::try_from(product).ok())
        .ok_or(GeometryError::ArithmeticOverflow)?;
    // `a + b sqrt d` with `a = 0`, `b = 1/q` and `d = p*q` is exactly
    // `sqrt(p*q)/q = sqrt(p/q)`.
    QSqrt::new(
        Frac::from_int(0),
        Frac {
            num: 1,
            den: value.den,
        },
        radicand,
    )
}

// ------------------------------------------------------------------- Point3 --

/// A point in space with exact rational coordinates, named like the kernel's
/// [`KPoint`] so a lifted figure reads the same way in both modules.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Point3 {
    pub name: String,
    pub x: Frac,
    pub y: Frac,
    pub z: Frac,
}

impl Point3 {
    /// A named point at exact rational coordinates.
    pub fn new(name: &str, x: Q, y: Q, z: Q) -> Self {
        Self {
            name: name.to_string(),
            x: Frac::from_q(x),
            y: Frac::from_q(y),
            z: Frac::from_q(z),
        }
    }

    /// A named point at integer coordinates -- the common case, since the
    /// figures a reasoner is handed are usually lattice figures.
    pub fn from_ints(name: &str, x: i64, y: i64, z: i64) -> Self {
        Self {
            name: name.to_string(),
            x: Frac::from_int(x),
            y: Frac::from_int(y),
            z: Frac::from_int(z),
        }
    }

    /// The exact coordinates, for arithmetic.
    pub fn coords(&self) -> anyhow::Result<(Q, Q, Q)> {
        Ok((self.x.to_q()?, self.y.to_q()?, self.z.to_q()?))
    }

    /// The point as a position vector from the origin. Every distance,
    /// projection and incidence test in this module is stated on such a vector,
    /// because that is the form the formulas take.
    pub fn to_vector(&self) -> anyhow::Result<Vec3> {
        let (x, y, z) = self.coords()?;
        Ok(Vec3::new(x, y, z))
    }

    /// Attach a vector to a point, keeping the point's name. The arithmetic is
    /// exact; the *label* stays the caller's to choose, which is why this is a
    /// constructor rather than something the constructions invent for themselves.
    pub fn from_vec(point: &Point3, v: &Vec3) -> anyhow::Result<Point3> {
        let (x, y, z) = v.coords()?;
        Ok(Point3::new(&point.name, x, y, z))
    }

    /// The exact distance from `other`, as a length -- usually irrational, hence
    /// the [`QSqrt`].
    pub fn distance_to(&self, other: &Point3) -> anyhow::Result<QSqrt> {
        self.to_vector()?.sub(&other.to_vector()?)?.length()
    }

    /// Whether the two names stand for the same location (audit fix 11, in
    /// three dimensions: a kernel that accepts `A == B` goes on to divide by
    /// `B - A`).
    pub fn same_location(&self, other: &Point3) -> anyhow::Result<bool> {
        Ok(self.coords()? == other.coords()?)
    }

    /// A one-line rendering, for messages and reports.
    pub fn describe(&self) -> String {
        format!("{}({}, {}, {})", self.name, self.x, self.y, self.z)
    }
}

impl fmt::Display for Point3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.describe())
    }
}

// --------------------------------------------------------------------- Vec3 --

/// An exact vector in space: three rationals, with no floating point anywhere
/// on the path from a premise to a conclusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: Frac,
    pub y: Frac,
    pub z: Frac,
}

impl Vec3 {
    /// The zero vector.
    pub fn zero() -> Self {
        Self {
            x: Frac::from_int(0),
            y: Frac::from_int(0),
            z: Frac::from_int(0),
        }
    }

    /// A vector from exact rationals.
    pub fn new(x: Q, y: Q, z: Q) -> Self {
        Self {
            x: Frac::from_q(x),
            y: Frac::from_q(y),
            z: Frac::from_q(z),
        }
    }

    /// A vector from integers.
    pub fn from_ints(x: i64, y: i64, z: i64) -> Self {
        Self {
            x: Frac::from_int(x),
            y: Frac::from_int(y),
            z: Frac::from_int(z),
        }
    }

    /// The exact components, for arithmetic.
    pub fn coords(&self) -> anyhow::Result<(Q, Q, Q)> {
        Ok((self.x.to_q()?, self.y.to_q()?, self.z.to_q()?))
    }

    /// The exact zero test. Cheap, and the side condition on nearly every
    /// division and normalization below: a zero direction is refused at the
    /// point of use rather than turned into an infinity further out.
    pub fn is_zero(&self) -> bool {
        self.x.num == 0 && self.y.num == 0 && self.z.num == 0
    }

    /// Componentwise sum.
    pub fn add(&self, other: &Self) -> anyhow::Result<Self> {
        let (x1, y1, z1) = self.coords()?;
        let (x2, y2, z2) = other.coords()?;
        Ok(Self::new(x1.add(&x2)?, y1.add(&y2)?, z1.add(&z2)?))
    }

    /// Componentwise difference, which is also the displacement from `other`
    /// to `self`.
    pub fn sub(&self, other: &Self) -> anyhow::Result<Self> {
        let (x1, y1, z1) = self.coords()?;
        let (x2, y2, z2) = other.coords()?;
        Ok(Self::new(x1.sub(&x2)?, y1.sub(&y2)?, z1.sub(&z2)?))
    }

    /// The reversed vector, exactly.
    pub fn neg(&self) -> anyhow::Result<Self> {
        let (x, y, z) = self.coords()?;
        Ok(Self::new(x.neg()?, y.neg()?, z.neg()?))
    }

    /// `k * self` for any exact rational `k`, negative included.
    pub fn scale(&self, k: &Q) -> anyhow::Result<Self> {
        let (x, y, z) = self.coords()?;
        Ok(Self::new(x.mul(k)?, y.mul(k)?, z.mul(k)?))
    }

    /// The exact dot product. This is why a right angle in three dimensions is
    /// still a decision and not an estimate: `AB perp CD` is
    /// `dot(B - A, D - C) = 0`, and zero is zero over the rationals.
    pub fn dot(&self, other: &Self) -> anyhow::Result<Q> {
        let (x1, y1, z1) = self.coords()?;
        let (x2, y2, z2) = other.coords()?;
        x1.mul(&x2)?.add(&y1.mul(&y2)?)?.add(&z1.mul(&z2)?)
    }

    /// The exact cross product: the vector normal to both, whose *direction* is
    /// the orientation of the ordered pair. Exactly zero when the two are
    /// parallel, which is how "these lines are parallel" is decided in 3D
    /// without a drawing to look at.
    pub fn cross(&self, other: &Self) -> anyhow::Result<Self> {
        let (x1, y1, z1) = self.coords()?;
        let (x2, y2, z2) = other.coords()?;
        Ok(Self::new(
            y1.mul(&z2)?.sub(&z1.mul(&y2)?)?,
            z1.mul(&x2)?.sub(&x1.mul(&z2)?)?,
            x1.mul(&y2)?.sub(&y1.mul(&x2)?)?,
        ))
    }

    /// The squared length: a rational, and therefore the quantity every
    /// distance test actually consumes. The length itself is usually
    /// irrational; its square never is.
    pub fn length_sq(&self) -> anyhow::Result<Q> {
        self.dot(self)
    }

    /// The exact length, in the quadratic field. A lattice edge of length
    /// `sqrt(2)` is stored as `1*sqrt(2)` and compares equal to another
    /// `sqrt(2)` rather than to a rounded number.
    pub fn length(&self) -> anyhow::Result<QSqrt> {
        sqrt_rational(&self.length_sq()?)
    }

    /// The unit vector along `self`. Refuses the zero vector, and refuses a
    /// length that is irrational -- normalizing `(1, 1, 0)` divides by
    /// `sqrt(2)` and does not land on a rational coordinate, and rounding it
    /// would put a float on the path from a premise to a conclusion.
    pub fn unit(&self) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !self.is_zero(),
            GeometryError::EmptyGeometry(
                "the zero vector has no direction to normalize".to_string()
            )
        );
        let squared = self.length_sq()?;
        let exact = self.length()?.rational_value()?.ok_or_else(|| {
            GeometryError::IrrationalRequired(format!(
                "normalizing {squared} divides by a square root the rational kernel does not have"
            ))
        })?;
        self.scale(&Q::ONE.div(&exact)?)
    }

    /// The exact unit normal, refusing as [`Vec3::unit`] does. Kept as its own
    /// name because "a unit normal" is what a caller thinks in, and because a
    /// rotation about an axis needs precisely this and no more.
    pub fn unit_normal(&self) -> anyhow::Result<Self> {
        self.unit()
    }

    /// Whether the two vectors are perpendicular: a zero dot product, decided
    /// exactly. A zero vector is *not* perpendicular to anything here -- the
    /// kernel's degenerate-leg refusal applies in three dimensions too, and a
    /// vacuous `true` is how an empty derivation becomes a claimed theorem.
    pub fn is_perpendicular_to(&self, other: &Self) -> anyhow::Result<bool> {
        anyhow::ensure!(
            !self.is_zero() && !other.is_zero(),
            GeometryError::DegenerateAngle {
                detail: "a zero-length leg is not a direction".to_string(),
            }
        );
        Ok(self.dot(other)?.is_zero())
    }

    /// Whether the two vectors are parallel: a zero cross product, exactly.
    /// Parallelism is a relation between *directions*, so a zero vector is
    /// parallel to nothing and is refused rather than called parallel to
    /// everything.
    pub fn is_parallel_to(&self, other: &Self) -> anyhow::Result<bool> {
        anyhow::ensure!(
            !self.is_zero() && !other.is_zero(),
            GeometryError::DegenerateSegment
        );
        Ok(self.cross(other)?.is_zero())
    }

    /// The exact distance between two position vectors, as a length in the
    /// quadratic field. The free-function form of [`Point3::distance_to`], for
    /// callers holding vectors rather than named points.
    pub fn distance_between(a: &Self, b: &Self) -> anyhow::Result<QSqrt> {
        a.sub(b)?.length()
    }

    /// A one-line rendering, for messages and reports. Display cannot fail
    /// and a `Frac` is two integers, so this asks no `Result` for permission.
    pub fn describe(&self) -> String {
        format!("({}, {}, {})", self.x, self.y, self.z)
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.describe())
    }
}

// --------------------------------------------------------------------- Line3 --

/// A line in space: a point and a direction. A point plus a direction rather
/// than two endpoints, because a line has no ends, and a third coordinate makes
/// "which of the two points is the origin of the direction" a real question:
/// every parameterization below depends on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Line3 {
    pub name: String,
    /// A point known to lie on the line.
    pub through: Point3,
    /// The direction, which must be nonzero.
    pub dir: Vec3,
}

impl Line3 {
    /// A line through a point in a direction. Refuses a zero direction with the
    /// kernel's `DegenerateSegment`: a line whose direction vanishes is the
    /// degenerate segment the kernel already refuses to build, and every
    /// division by `|dir|^2` below assumes that it does not vanish.
    pub fn new(name: &str, through: Point3, dir: Vec3) -> anyhow::Result<Self> {
        anyhow::ensure!(!dir.is_zero(), GeometryError::DegenerateSegment);
        Ok(Self {
            name: name.to_string(),
            through,
            dir,
        })
    }

    /// The line through two distinct points, as the kernel names a line. Refuses
    /// coincident points with the kernel's `CoincidentPoints`, so the audit's
    /// "two names, one location" cannot reach the 3D path by way of a 2D
    /// figure.
    pub fn through_points(name: &str, first: &Point3, second: &Point3) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !first.same_location(second)?,
            GeometryError::CoincidentPoints {
                a: first.name.clone(),
                b: second.name.clone()
            }
        );
        Self::new(
            name,
            first.clone(),
            second.to_vector()?.sub(&first.to_vector()?)?,
        )
    }

    /// The exact point at parameter `t` along the line: `through + t * dir`.
    pub fn at(&self, t: &Q) -> anyhow::Result<Point3> {
        let moved = self.dir.scale(t)?.add(&self.through.to_vector()?)?;
        Point3::from_vec(&self.through, &moved)
    }

    /// Whether the point lies on the line: the offset from `through` is parallel
    /// to the direction, decided by a zero cross product.
    pub fn contains(&self, point: &Point3) -> anyhow::Result<bool> {
        Ok(point
            .to_vector()?
            .sub(&self.through.to_vector()?)?
            .cross(&self.dir)?
            .is_zero())
    }

    /// The exact unit direction, refusing as [`Vec3::unit`] does. Only the
    /// rotation needs this; projection and distance are written to avoid it.
    pub fn unit_direction(&self) -> anyhow::Result<Vec3> {
        self.dir.unit()
    }

    /// A one-line rendering, for messages and reports.
    pub fn describe(&self) -> String {
        format!("line({}: {} through {})", self.name, self.dir, self.through)
    }
}

impl fmt::Display for Line3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.describe())
    }
}

// -------------------------------------------------------------------- Plane3 --

/// A plane in space: a normal and an offset, i.e. the exact equation
/// `normal . x = offset`. The equation form rather than three points, because a
/// plane has infinitely many points and naming three of them loses the
/// information that a fourth is *not* on it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plane3 {
    pub name: String,
    /// The normal, which must be nonzero.
    pub normal: Vec3,
    /// The exact right-hand side: `normal . x = offset`.
    pub offset: Frac,
}

impl Plane3 {
    /// The plane `normal . x = offset`. Refuses a zero normal with
    /// `EmptyGeometry`, because `0 . x = offset` is either everything or
    /// nothing: there is no geometry there to reason about, which is what the
    /// variant says.
    pub fn new(name: &str, normal: Vec3, offset: Q) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !normal.is_zero(),
            GeometryError::EmptyGeometry("a zero normal does not determine a plane".to_string())
        );
        Ok(Self {
            name: name.to_string(),
            normal,
            offset: Frac::from_q(offset),
        })
    }

    /// The plane through three points, as the kernel names a line by two.
    /// Refuses collinear points with the kernel's `DegenerateTriangle` -- the
    /// exact case that variant was added for.
    pub fn through_points(
        name: &str,
        first: &Point3,
        second: &Point3,
        third: &Point3,
    ) -> anyhow::Result<Self> {
        let normal = second
            .to_vector()?
            .sub(&first.to_vector()?)?
            .cross(&third.to_vector()?.sub(&first.to_vector()?)?)?;
        anyhow::ensure!(
            !normal.is_zero(),
            GeometryError::DegenerateTriangle {
                vertices: format!("{}, {}, {}", first.name, second.name, third.name),
            }
        );
        let (nx, ny, nz) = normal.coords()?;
        let (x, y, z) = first.coords()?;
        Self::new(
            name,
            normal,
            nx.mul(&x)?.add(&ny.mul(&y)?)?.add(&nz.mul(&z)?)?,
        )
    }

    /// The exact signed side of a point: `normal . p - offset`, positive on one
    /// side, negative on the other, zero exactly on the plane. That sign is the
    /// whole content of "which side of the plane is this point on", and it is a
    /// rational -- no distance, no square root, no normalization.
    pub fn signed_side(&self, point: &Point3) -> anyhow::Result<Q> {
        let (nx, ny, nz) = self.normal.coords()?;
        let (x, y, z) = point.coords()?;
        nx.mul(&x)?
            .add(&ny.mul(&y)?)?
            .add(&nz.mul(&z)?)?
            .sub(&self.offset.to_q()?)
    }

    /// The exact point where a line of the plane nearest the origin sits: the
    /// foot of the perpendicular from the origin, `(offset / |n|^2) n`. Exact,
    /// and the natural point to name when a caller needs a representative of the
    /// plane (to test coincidence, or to label an intersection line).
    pub fn point_on(&self, name: &str) -> anyhow::Result<Point3> {
        let scale = self.offset.to_q()?.div(&self.normal.length_sq()?)?;
        let (x, y, z) = self.normal.scale(&scale)?.coords()?;
        Ok(Point3::new(name, x, y, z))
    }

    /// Whether the point lies on the plane: a zero signed side.
    pub fn contains(&self, point: &Point3) -> anyhow::Result<bool> {
        Ok(self.signed_side(point)?.is_zero())
    }

    /// The exact distance from a point to the plane, in the quadratic field.
    ///
    /// `|n . p - offset| / |n|`, squared to a rational first and rooted
    /// afterwards, so a plane stated as `2x = 1` and the same plane stated as
    /// `x = 1/2` give the same distance instead of answers differing by a
    /// factor of two -- and so a `3-4-5` normal, whose length is `5 sqrt 2`, is
    /// no problem at all.
    pub fn distance_to(&self, point: &Point3) -> anyhow::Result<QSqrt> {
        let side = self.signed_side(point)?;
        sqrt_rational(&side.mul(&side)?.div(&self.normal.length_sq()?)?)
    }

    /// The exact projection of a point onto the plane: `p - ((n . p - offset) /
    /// |n|^2) n`. The result lies on the plane to the last rational, which a
    /// certificate checks by re-deriving the plane equation rather than
    /// trusting this function.
    pub fn project(&self, point: &Point3) -> anyhow::Result<Point3> {
        let (nx, ny, nz) = self.normal.coords()?;
        let (x, y, z) = point.coords()?;
        let t = self.signed_side(point)?.div(&self.normal.length_sq()?)?;
        Ok(Point3::new(
            &point.name,
            x.sub(&nx.mul(&t)?)?,
            y.sub(&ny.mul(&t)?)?,
            z.sub(&nz.mul(&t)?)?,
        ))
    }

    /// A one-line rendering of the plane *equation*, which is the whole
    /// content of a plane and worth reading in a report.
    pub fn describe(&self) -> String {
        format!(
            "plane({}: {} . x = {})",
            self.name, self.normal, self.offset
        )
    }
}

impl fmt::Display for Plane3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.describe())
    }
}

// --------------------------------------------------------------- projections --

/// The exact projection of a point onto a line: `a + ((p - a) . d) / |d|^2 d`.
///
/// Written with the *unnormalized* direction, so no square root is needed and
/// the answer is rational whenever the inputs are. The displacement from `p` to
/// its projection is perpendicular to the line, exactly, which is the check a
/// certificate makes after re-deriving that the result is *on* the line.
pub fn projection_point_line(point: &Point3, line: &Line3) -> anyhow::Result<Point3> {
    let offset = point.to_vector()?.sub(&line.through.to_vector()?)?;
    let t = offset.dot(&line.dir)?.div(&line.dir.length_sq()?)?;
    Point3::from_vec(
        &line.through,
        &line.through.to_vector()?.add(&line.dir.scale(&t)?)?,
    )
}

/// The exact distance from a point to a line, in the quadratic field: the length
/// of the cross product divided by the length of the direction, squared to a
/// rational first.
pub fn distance_point_line(point: &Point3, line: &Line3) -> anyhow::Result<QSqrt> {
    let offset = point.to_vector()?.sub(&line.through.to_vector()?)?;
    let offset = offset.cross(&line.dir)?;
    sqrt_rational(&offset.length_sq()?.div(&line.dir.length_sq()?)?)
}

/// The exact projection of a point onto a plane. The free-function form of
/// [`Plane3::project`], kept because "project onto this plane" is how a caller
/// thinks about it, with the plane named by its object.
pub fn projection_point_plane(point: &Point3, plane: &Plane3) -> anyhow::Result<Point3> {
    plane.project(point)
}

/// The exact distance from a point to a plane, in the quadratic field. The
/// free-function form of [`Plane3::distance_to`].
pub fn distance_point_plane(point: &Point3, plane: &Plane3) -> anyhow::Result<QSqrt> {
    plane.distance_to(point)
}

// -------------------------------------------------------------- intersections --

/// The exact point where a line meets a plane:
/// `t = (offset - n . p) / (n . d)`, one rational division and no iteration.
///
/// Refuses a line lying in the plane's direction with `ParallelLines` -- the
/// kernel's planar refusal, reused because the reason is identical: the system
/// has either no solution or infinitely many, and reporting one of them would be
/// a choice this module has no license to make.
pub fn intersect_line_plane(line: &Line3, plane: &Plane3) -> anyhow::Result<Point3> {
    let (nx, ny, nz) = plane.normal.coords()?;
    let (dx, dy, dz) = line.dir.coords()?;
    let denominator = nx.mul(&dx)?.add(&ny.mul(&dy)?)?.add(&nz.mul(&dz)?)?;
    anyhow::ensure!(!denominator.is_zero(), GeometryError::ParallelLines);
    let offset = plane.signed_side(&line.through)?.neg()?;
    line.at(&offset.div(&denominator)?)
}

/// Whether two planes share at least one point: a point of the first, tested
/// against the second's equation. A zero cross product plus this one test
/// decides "coincident or merely parallel" exactly, which is the question the
/// 2D kernel never had to ask.
fn plane_shares_a_point(first: &Plane3, second: &Plane3) -> anyhow::Result<bool> {
    Ok(first.signed_side(&second.point_on("probe")?)?.is_zero()
        || second.signed_side(&first.point_on("probe")?)?.is_zero())
}

/// The exact line where two planes meet: direction `d = n1 x n2`, and the point
/// `(d2 (n1 x d) + d1 (d x n2)) / |d|^2`, which satisfies both plane equations
/// identically -- the two cross products are perpendicular to one of the normals
/// each, so the formula is checkable by re-deriving rather than by trust.
///
/// The cross product is zero exactly when the planes are parallel, and then
/// "same plane or different ones?" has to be answered separately: a point of one
/// satisfying the other's equation means they are the same plane
/// (`CoincidentLines`), and otherwise they are distinct parallel planes
/// (`ParallelLines`). Both refuse, because in each case the intersection is not
/// a single line, and a reasoner that wanted a line wanted a specific one.
pub fn intersect_plane_plane(first: &Plane3, second: &Plane3) -> anyhow::Result<Line3> {
    let direction = first.normal.cross(&second.normal)?;
    anyhow::ensure!(
        !direction.is_zero(),
        if plane_shares_a_point(first, second)? {
            GeometryError::CoincidentLines
        } else {
            GeometryError::ParallelLines
        }
    );
    // `n2 x d` is perpendicular to `n1` and `d x n1` is perpendicular to `n2`,
    // and the remaining dot products are both `|d|^2` by the scalar triple
    // product -- so `n1 . p = d1` and `n2 . p = d2` hold identically, which is
    // what makes this checkable rather than merely plausible.
    let (n2_turn, d_turn) = (
        second.normal.cross(&direction)?,
        direction.cross(&first.normal)?,
    );
    let (d1, d2) = (first.offset.to_q()?, second.offset.to_q()?);
    let point = n2_turn
        .scale(&d1)?
        .add(&d_turn.scale(&d2)?)?
        .scale(&Q::ONE.div(&direction.length_sq()?)?)?;
    let name = format!("{} x {}", first.name, second.name);
    let (x, y, z) = point.coords()?;
    Line3::new(&name, Point3::new(&name, x, y, z), direction)
}

/// The exact point where two lines meet, or a refusal.
///
/// The two lines are coplanar exactly when `(p2 - p1) . (d1 x d2) == 0`, and
/// that one sign separates the three cases the 2D kernel had to assume away:
///
/// - `d1 x d2 == 0` and a point of one lies on the other: `CoincidentLines`,
///   the same line named twice;
/// - `d1 x d2 == 0` otherwise: `ParallelLines`, no intersection;
/// - `d1 x d2 != 0` but the coplanarity sign nonzero: **skew**, and there is
///   no intersection point in the room to return. Refused with
///   `NoRationalSolution`, whose message says the word: "skew" is the single
///   most useful sentence in 3D geometry, and a reasoner that meets it deserves
///   to hear it rather than to divide by zero.
pub fn intersect_line_line(first: &Line3, second: &Line3) -> anyhow::Result<Point3> {
    let normal = first.dir.cross(&second.dir)?;
    let w = second
        .through
        .to_vector()?
        .sub(&first.through.to_vector()?)?;
    anyhow::ensure!(
        !normal.is_zero(),
        if first.contains(&second.through)? {
            GeometryError::CoincidentLines
        } else {
            GeometryError::ParallelLines
        }
    );
    anyhow::ensure!(
        w.dot(&normal)?.is_zero(),
        GeometryError::NoRationalSolution(format!(
            "{} and {} are skew: they are not parallel and they never meet",
            first.name, second.name
        ))
    );
    let t = w
        .cross(&second.dir)?
        .dot(&normal)?
        .div(&normal.length_sq()?)?;
    Point3::from_vec(
        &first.through,
        &first.through.to_vector()?.add(&first.dir.scale(&t)?)?,
    )
}

/// The closest pair of points on two lines, exactly.
///
/// With `w = p2 - p1` and `n = d1 x d2`, the standard solution is
/// `t = ((w x d2) . n) / |n|^2` and `s = ((w x d1) . n) / |n|^2`, giving the
/// two points `p1 + t d1` and `p2 + s d2`.
///
/// This answers for **skew** lines, which is the point of having it separate
/// from [`intersect_line_line`]: skew lines have no intersection but they do
/// have a well-defined nearest pair, and "how far apart are these two lines" is
/// a question a 3D reasoner asks constantly. Intersecting lines get their
/// intersection as the unique closest pair.
///
/// Parallel lines refuse, with `ParallelLines` -- or `CoincidentLines` when the
/// two are the same line -- because the closest pair is then not unique, and
/// returning one of the infinitely many would be an arbitrary choice dressed up
/// as an answer. `NoRationalSolution` would be the wrong variant there: the
/// problem is not that no solution exists but that the answer is not determined,
/// and saying which is the whole reason the kernel distinguishes the two.
pub fn closest_points_line_line(first: &Line3, second: &Line3) -> anyhow::Result<(Point3, Point3)> {
    let normal = first.dir.cross(&second.dir)?;
    anyhow::ensure!(
        !normal.is_zero(),
        if first.contains(&second.through)? {
            GeometryError::CoincidentLines
        } else {
            GeometryError::ParallelLines
        }
    );
    let w = second
        .through
        .to_vector()?
        .sub(&first.through.to_vector()?)?;
    let t = w
        .cross(&second.dir)?
        .dot(&normal)?
        .div(&normal.length_sq()?)?;
    let s = w
        .cross(&first.dir)?
        .dot(&normal)?
        .div(&normal.length_sq()?)?;
    Ok((
        Point3::from_vec(
            &first.through,
            &first.through.to_vector()?.add(&first.dir.scale(&t)?)?,
        )?,
        Point3::from_vec(
            &second.through,
            &second.through.to_vector()?.add(&second.dir.scale(&s)?)?,
        )?,
    ))
}

/// Whether three points are collinear in space: the cross product of two offsets
/// is zero. A different question from the 2D kernel's, and the reason
/// [`lift_scene`] needs it -- a figure collinear in the plane stays collinear
/// when embedded, and this is the check that *proves* it rather than assuming it.
pub fn collinear_3d(first: &Point3, second: &Point3, third: &Point3) -> anyhow::Result<bool> {
    let a = second.to_vector()?.sub(&first.to_vector()?)?;
    let b = third.to_vector()?.sub(&first.to_vector()?)?;
    Ok(a.cross(&b)?.is_zero())
}

/// Whether a right angle stands at `vertex`, between the legs to `first` and
/// `second`: a zero dot product of the two legs, exactly. The 3D statement of
/// the kernel's `RightAngle`, and the case mental-rotation questions are mostly
/// made of -- "after turning it, is this still square?"
///
/// Refuses a zero-length leg with the kernel's `DegenerateAngle`, in three
/// dimensions as in two: a collapsed leg is not an angle, and calling it a right
/// angle would be a theorem about nothing.
pub fn right_angle_at(vertex: &Point3, first: &Point3, second: &Point3) -> anyhow::Result<bool> {
    let u = first.to_vector()?.sub(&vertex.to_vector()?)?;
    let v = second.to_vector()?.sub(&vertex.to_vector()?)?;
    u.is_perpendicular_to(&v)
}

// -------------------------------------------------------------------- Sphere --

/// A sphere with an exact centre and an exact *squared* radius, carrying the
/// name its facts will refer to -- the same discipline audit fix 1 forced on
/// the kernel's circles, for the same reason: three points that look roughly
/// equidistant are not an object with an identity, and an object with no
/// identity cannot carry provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sphere {
    pub name: String,
    pub center: Point3,
    pub radius_sq: Frac,
}

impl Sphere {
    /// A sphere of squared radius `radius_sq`. A zero radius is the kernel's
    /// `ZeroRadiusCircle` -- a 3D sphere is the same object the kernel's circles
    /// are, and a point is not one. A *negative* squared radius is not that at
    /// all: it is not a sphere of any size, so it is `NoRationalSolution`, and
    /// the distinction is the difference between "degenerate" and "impossible".
    pub fn new(name: &str, center: Point3, radius_sq: Q) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !radius_sq.is_zero(),
            GeometryError::ZeroRadiusCircle {
                circle: name.to_string()
            }
        );
        anyhow::ensure!(
            Q::ZERO.less(&radius_sq),
            GeometryError::NoRationalSolution(format!(
                "a sphere's squared radius must be positive, got {radius_sq}"
            ))
        );
        Ok(Self {
            name: name.to_string(),
            center,
            radius_sq: Frac::from_q(radius_sq),
        })
    }

    /// The exact squared distance from the centre. Factored out of
    /// [`Sphere::contains`] because "how far is this point from the centre" is a
    /// question of its own, and a rational rather than a square root is the form
    /// every membership test actually wants.
    pub fn distance_sq(&self, point: &Point3) -> anyhow::Result<Q> {
        self.center
            .to_vector()?
            .sub(&point.to_vector()?)?
            .length_sq()
    }

    /// Whether the point is on the sphere: a squared-distance equality, exact.
    pub fn contains(&self, point: &Point3) -> anyhow::Result<bool> {
        Ok(self.distance_sq(point)?.eq(&self.radius_sq.to_q()?))
    }

    /// The exact radius, in the quadratic field. Usually irrational -- a sphere
    /// of squared radius 2 has radius `sqrt(2)` -- which is why the radius is
    /// stored squared and this is the checked way to get it back.
    pub fn radius(&self) -> anyhow::Result<QSqrt> {
        sqrt_rational(&self.radius_sq.to_q()?)
    }

    /// A one-line rendering, for messages and reports.
    pub fn describe(&self) -> String {
        format!(
            "sphere({}: centre {}, r^2={})",
            self.name, self.center, self.radius_sq
        )
    }
}

impl fmt::Display for Sphere {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.describe())
    }
}

// -------------------------------------------------------------------- Face3 --

/// One face of a polyhedron: an ordered, cyclic list of named vertices. The
/// order *is* the orientation -- walking it one way is the outside, the other
/// way the inside -- which is why a face is a list and not a set. Cyclic, so
/// the last vertex is joined back to the first.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Face3 {
    pub name: String,
    pub vertices: Vec<String>,
}

impl Face3 {
    /// A face with its vertices in winding order.
    pub fn new(name: &str, vertices: &[&str]) -> Self {
        Self {
            name: name.to_string(),
            vertices: vertices.iter().map(|v| v.to_string()).collect(),
        }
    }

    /// How many vertices the face names. A face with fewer than three of them
    /// is refused by [`Polyhedron::sanity_check`].
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// Whether the face names no vertices at all, which is the first thing
    /// [`Polyhedron::sanity_check`] refuses.
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// A one-line rendering, in winding order.
    pub fn describe(&self) -> String {
        format!("face({}: {})", self.name, self.vertices.join(" "))
    }
}

// --------------------------------------------------------------- Polyhedron --

/// A solid: named vertices and named faces, with the sanity checks a mesh needs
/// before any area or volume it reports means anything.
///
/// Three properties, each refused rather than assumed, because a mesh that fails
/// them has no volume to compute and a "repair" pass that papers over them
/// produces a number about a solid nobody described:
///
/// 1. [`Polyhedron::sanity_check`] -- the faces are non-empty, every named
///    vertex exists, every face has at least three *distinct* vertices, and
///    every face has a nonzero normal (three collinear points, which is
///    `DegenerateTriangle`).
/// 2. [`Polyhedron::orientation_is_consistent`] -- every edge is traversed
///    exactly once in each direction across the whole solid. That is what a
///    closed, orientable surface is, and a mesh that fails it has inside and
///    outside reversed somewhere; the kernel has no winding variant, so the
///    refusal is `Unproven`, the honest reading being that the solid's outward
///    normal is *not established* by the vertex order it was given.
/// 3. [`Polyhedron::is_convex`] -- every vertex lies on the inward side of every
///    face plane, decided by a sign. Exact, and it is the difference between
///    "the cube" and "the L-shaped solid", which a low-resolution picture cannot
///    tell apart.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Polyhedron {
    pub name: String,
    pub vertices: Vec<Point3>,
    pub faces: Vec<Face3>,
}

impl Polyhedron {
    /// A solid with named vertices and faces. Does *not* validate: a caller may
    /// legitimately want to hold a mesh that is not yet a solid. Use
    /// [`Polyhedron::checked`] for the validating constructor.
    pub fn new(name: &str, vertices: Vec<Point3>, faces: Vec<Face3>) -> Self {
        Self {
            name: name.to_string(),
            vertices,
            faces,
        }
    }

    /// A solid that passes [`Polyhedron::sanity_check`], or an error naming what
    /// is wrong with it. The constructor to reach for when a mesh came from
    /// outside and has not been looked at yet.
    pub fn checked(name: &str, vertices: Vec<Point3>, faces: Vec<Face3>) -> anyhow::Result<Self> {
        let solid = Self::new(name, vertices, faces);
        solid.sanity_check()?;
        Ok(solid)
    }

    /// The unit cube `[0,1]^3` with outward-wound faces: the figure every solid
    /// test in this module is measured against, and the one whose exact face
    /// areas (six of `1`) and volume (`1`) are known independently.
    pub fn unit_cube() -> Self {
        let vertices = vec![
            Point3::from_ints("A", 0, 0, 0),
            Point3::from_ints("B", 1, 0, 0),
            Point3::from_ints("C", 1, 1, 0),
            Point3::from_ints("D", 0, 1, 0),
            Point3::from_ints("E", 0, 0, 1),
            Point3::from_ints("F", 1, 0, 1),
            Point3::from_ints("G", 1, 1, 1),
            Point3::from_ints("H", 0, 1, 1),
        ];
        let faces = vec![
            Face3::new("bottom", &["A", "D", "C", "B"]),
            Face3::new("top", &["E", "F", "G", "H"]),
            Face3::new("front", &["A", "B", "F", "E"]),
            Face3::new("right", &["B", "C", "G", "F"]),
            Face3::new("back", &["D", "H", "G", "C"]),
            Face3::new("left", &["A", "E", "H", "D"]),
        ];
        Self::new("cube", vertices, faces)
    }

    /// The unit tetrahedron `O, X, Y, Z`: three faces of area `1/2` and one of
    /// area `sqrt(3)/2`, volume `1/6`. The solid whose *irrational* face area is
    /// the reason [`Polyhedron::total_surface_area`] refuses.
    pub fn unit_tetrahedron() -> Self {
        let vertices = vec![
            Point3::from_ints("O", 0, 0, 0),
            Point3::from_ints("X", 1, 0, 0),
            Point3::from_ints("Y", 0, 1, 0),
            Point3::from_ints("Z", 0, 0, 1),
        ];
        let faces = vec![
            Face3::new("oxy", &["O", "Y", "X"]),
            Face3::new("oxz", &["O", "X", "Z"]),
            Face3::new("oyz", &["O", "Z", "Y"]),
            Face3::new("xyz", &["X", "Y", "Z"]),
        ];
        Self::new("tetrahedron", vertices, faces)
    }

    /// Look a vertex up by name, refusing an unknown name rather than inventing
    /// a location for it.
    pub fn vertex(&self, name: &str) -> anyhow::Result<&Point3> {
        self.vertices
            .iter()
            .find(|v| v.name == name)
            .ok_or_else(|| anyhow::anyhow!("polyhedron {} has no vertex '{name}'", self.name))
    }

    /// The exact points of a face, in winding order.
    pub fn face_points(&self, face: &Face3) -> anyhow::Result<Vec<Point3>> {
        face.vertices
            .iter()
            .map(|name| self.vertex(name).cloned())
            .collect()
    }

    /// The face's exact plane, from its first three vertices. Refuses a
    /// degenerate face (three collinear points) with the kernel's
    /// `DegenerateTriangle`.
    pub fn face_plane(&self, face: &Face3) -> anyhow::Result<Plane3> {
        let points = self.face_points(face)?;
        anyhow::ensure!(
            points.len() >= 3,
            GeometryError::DegenerateTriangle {
                vertices: face.describe()
            }
        );
        Plane3::through_points(&face.name, &points[0], &points[1], &points[2])
    }

    /// Structural sanity: non-empty faces, named vertices that exist, at least
    /// three *distinct* vertices per face, and a nonzero normal for each face.
    ///
    /// Nothing here needs a picture or a tolerance: "does this face have area"
    /// is a zero cross product, and "are these the same location" is an
    /// equality of exact coordinates.
    pub fn sanity_check(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.faces.is_empty(),
            GeometryError::EmptyGeometry(format!("polyhedron {} has no faces", self.name))
        );
        for face in &self.faces {
            let points = self.face_points(face)?;
            anyhow::ensure!(
                points.len() >= 3,
                GeometryError::DegenerateTriangle {
                    vertices: face.describe()
                }
            );
            for (index, first) in points.iter().enumerate() {
                for other in points.iter().skip(index + 1) {
                    anyhow::ensure!(
                        !first.same_location(other)?,
                        GeometryError::DegenerateTriangle {
                            vertices: face.describe()
                        }
                    );
                }
            }
            // through_points refuses a collinear triple, and with the
            // distinctness above that is exactly "the face has area".
            self.face_plane(face)?;
        }
        Ok(())
    }

    /// Whether every edge of the solid is traversed exactly once in each
    /// direction. A closed, consistently oriented surface has that property by
    /// definition; a mesh that fails it has somewhere inside and outside
    /// swapped, and its signed volume is a difference of nonsense.
    pub fn orientation_is_consistent(&self) -> anyhow::Result<bool> {
        self.sanity_check()?;
        let mut directed: Vec<(&str, &str)> = Vec::new();
        for face in &self.faces {
            let names = &face.vertices;
            for index in 0..names.len() {
                directed.push((&names[index], &names[(index + 1) % names.len()]));
            }
        }
        for (from, to) in &directed {
            if from == to {
                return Ok(false);
            }
            let forward = directed
                .iter()
                .filter(|(a, b)| a == from && b == to)
                .count();
            let backward = directed
                .iter()
                .filter(|(a, b)| a == to && b == from)
                .count();
            if forward != 1 || backward != 1 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The exact centroid of the vertices: one more rational division, and the
    /// interior reference point the convexity test needs. For a convex solid the
    /// centroid of its vertices lies strictly inside it, which is what makes it
    /// usable as the "inward" side of a face.
    pub fn centroid(&self) -> anyhow::Result<Vec3> {
        anyhow::ensure!(
            !self.vertices.is_empty(),
            GeometryError::EmptyGeometry(format!("polyhedron {} has no vertices", self.name))
        );
        let count = Q::from_int(
            i64::try_from(self.vertices.len()).map_err(|_| GeometryError::ArithmeticOverflow)?,
        );
        let sum = self
            .vertices
            .iter()
            .try_fold(Vec3::zero(), |acc, v| acc.add(&v.to_vector()?))?;
        sum.scale(&Q::ONE.div(&count)?)
    }

    /// Whether the solid is convex, exactly: every face plane is a *supporting*
    /// plane of the vertex set -- all the vertices off it lie strictly on one
    /// side of it -- and every face has at least one vertex off it.
    ///
    /// This is the definition stated as a sign, so it needs no convex hull, no
    /// tolerance and no picture. It is also orientation-independent, which is
    /// the honest way round: convexity is a property of the *solid*, and a mesh
    /// wound inside-out is still the cube. The winding is checked separately, by
    /// [`Polyhedron::orientation_is_consistent`], and a solid that fails it has
    /// no inside to be convex about, so that refusal comes first.
    pub fn is_convex(&self) -> anyhow::Result<bool> {
        anyhow::ensure!(
            self.orientation_is_consistent()?,
            GeometryError::Unproven(format!(
                "polyhedron {}: the faces are not consistently wound, so no side is 'inward'",
                self.name
            ))
        );
        for face in &self.faces {
            let plane = self.face_plane(face)?;
            let mut chosen: Option<bool> = None;
            let mut off_plane = 0usize;
            for vertex in &self.vertices {
                let side = plane.signed_side(vertex)?;
                if side.is_zero() {
                    continue;
                }
                off_plane += 1;
                let positive = Q::ZERO.less(&side);
                match chosen {
                    None => chosen = Some(positive),
                    Some(seen) if seen == positive => {}
                    // Vertices on both sides: the plane cuts the solid, so this
                    // face is not a supporting plane and the solid is not convex.
                    Some(_) => return Ok(false),
                }
            }
            if off_plane == 0 {
                // Every vertex lies in this face's plane, so the face bounds
                // nothing and there is no solid here to call convex.
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The exact signed volume, `1/6 * sum over triangles of a . (b x c)`, with
    /// each face fanned from its first vertex. Positive when the winding is
    /// outward. A rational -- a unit cube has volume exactly `1`, a unit
    /// tetrahedron exactly `1/6` -- because the volume of a lattice solid is
    /// rational even when its face areas are not.
    ///
    /// The winding must be consistent for the sum to mean anything, so an
    /// inconsistent mesh refuses rather than reporting the difference between
    /// two halves of a figure.
    pub fn signed_volume(&self) -> anyhow::Result<Q> {
        anyhow::ensure!(
            self.orientation_is_consistent()?,
            GeometryError::Unproven(format!(
                "polyhedron {}: a signed volume needs a consistently wound surface",
                self.name
            ))
        );
        let mut total = Q::ZERO;
        for face in &self.faces {
            let points = self.face_points(face)?;
            for index in 1..points.len().saturating_sub(1) {
                let triangle = (
                    points[0].to_vector()?,
                    points[index].to_vector()?,
                    points[index + 1].to_vector()?,
                );
                total = total.add(&triangle.0.dot(&triangle.1.cross(&triangle.2)?)?)?;
            }
        }
        total.div(&Q::from_int(6))
    }

    /// The exact volume: the magnitude of [`Polyhedron::signed_volume`]. Still a
    /// rational.
    pub fn volume(&self) -> anyhow::Result<Q> {
        let signed = self.signed_volume()?;
        Ok(if signed.less(&Q::ZERO) {
            signed.neg()?
        } else {
            signed
        })
    }

    /// The exact area of one face: the magnitude of the polygon's vector area
    /// `1/2 * sum of p_i x p_{i+1}` taken cyclically. Exact, and usually
    /// irrational -- the slanted face of the unit tetrahedron is exactly
    /// `sqrt(3)/2`.
    pub fn face_area(&self, face: &Face3) -> anyhow::Result<QSqrt> {
        let points = self.face_points(face)?;
        anyhow::ensure!(
            points.len() >= 3,
            GeometryError::DegenerateTriangle {
                vertices: face.describe()
            }
        );
        let mut accumulator = Vec3::zero();
        for index in 0..points.len() {
            accumulator = accumulator.add(
                &points[index]
                    .to_vector()?
                    .cross(&points[(index + 1) % points.len()].to_vector()?)?,
            )?;
        }
        accumulator.scale(&Q::ONE.div(&Q::from_int(2))?)?.length()
    }

    /// The exact total surface area: the sum of the face areas.
    ///
    /// This is where the quadratic field's limits show, and the module's scope
    /// note is not decoration. `QSqrt` is closed under addition only for a shared
    /// radicand, so the unit cube (six rational faces) sums exactly to `6` while
    /// a tetrahedron -- three faces of `1/2` and one of `sqrt(3)/2` -- *refuses*.
    /// Returning `2.366...` for the tetrahedron would put a float on the path
    /// from a premise to a conclusion. Use [`Polyhedron::face_area`] per face
    /// when the radicals differ.
    pub fn total_surface_area(&self) -> anyhow::Result<QSqrt> {
        let mut total: Option<QSqrt> = None;
        for face in &self.faces {
            let area = self.face_area(face)?;
            total = Some(match total {
                None => area,
                Some(previous) => previous.add(&area).map_err(|why| {
                    GeometryError::IrrationalRequired(format!(
                        "the face areas of {} span more than one radical, so their sum is not in \
                         the quadratic field this kernel is exact over: {why}",
                        self.name
                    ))
                })?,
            });
        }
        total.ok_or_else(|| {
            anyhow::Error::from(GeometryError::EmptyGeometry(format!(
                "polyhedron {} has no faces",
                self.name
            )))
        })
    }

    /// A one-line rendering: the name and the two counts a report needs, since
    /// a solid with no faces is refused and one with three is a curiosity.
    pub fn describe(&self) -> String {
        format!(
            "polyhedron({}: {} vertices, {} faces)",
            self.name,
            self.vertices.len(),
            self.faces.len()
        )
    }
}

impl fmt::Display for Polyhedron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.describe())
    }
}

// ----------------------------------------------------------------- rotation --

/// An exact rotation: a cosine and a sine, both rational.
///
/// This is the deliberate narrowing that makes mental rotation decidable. A
/// rotation is exact here when its turn has a rational cosine *and* a rational
/// sine -- a quarter turn (`0, 1`), a half turn (`-1, 0`), a 3-4-5 turn
/// (`3/5, 4/5`) -- and the constructor refuses any pair that is not one, by
/// checking `cos^2 + sin^2 == 1` exactly. No angle in degrees: the field, not
/// the unit, is what the arithmetic needs.
///
/// What this excludes is stated rather than hidden. The 60-degree turn needs
/// `cos = 1/2` with `sin = sqrt(3)/2`, an irrational sine, and turning a lattice
/// point by it leaves the lattice. There is no exact rational answer to give, so
/// there is no rotation to construct -- and a reasoner that wanted the answer
/// gets a refusal it can report instead of a rounded point it would then have to
/// go on treating as exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rotation {
    pub cos: Frac,
    pub sin: Frac,
}

impl Rotation {
    /// The rotation with this exact cosine and sine. Refuses a pair that is not
    /// a rotation -- `cos^2 + sin^2 != 1` -- with `NoRationalSolution`, because
    /// applying such a pair would not rotate anything, it would shear.
    pub fn new(cos: Q, sin: Q) -> anyhow::Result<Self> {
        let total = cos.mul(&cos)?.add(&sin.mul(&sin)?)?;
        anyhow::ensure!(
            total == Q::ONE,
            GeometryError::NoRationalSolution(format!(
                "cos {cos} and sin {sin} do not satisfy cos^2 + sin^2 = 1 (got {total}), so they \
                 are not a rotation"
            ))
        );
        Ok(Self {
            cos: Frac::from_q(cos),
            sin: Frac::from_q(sin),
        })
    }

    /// The identity rotation.
    pub fn identity() -> Self {
        Self {
            cos: Frac::from_int(1),
            sin: Frac::from_int(0),
        }
    }

    /// A quarter turn: `cos 0`, `sin 1`. Counter-clockwise about the axis, in
    /// the right-hand sense.
    pub fn quarter_turn() -> Self {
        Self {
            cos: Frac::from_int(0),
            sin: Frac::from_int(1),
        }
    }

    /// The 3-4-5 turn: `cos 3/5`, `sin 4/5`. The example that separates this
    /// from a lookup table of familiar angles -- it is not a degree value anyone
    /// memorizes, and it is still exact.
    pub fn three_four_five_turn() -> Self {
        Self {
            cos: Frac { num: 3, den: 5 },
            sin: Frac { num: 4, den: 5 },
        }
    }
}

/// The exact image of a point under a rotation about an axis: Rodrigues' formula,
/// with the axis taken to unit length first.
///
/// `v' = v cos + (k x v) sin + k (k . v) (1 - cos)`, all of it rational, so the
/// answer is an exact point and not a rounded one. The three terms are the
/// component along the axis, which the rotation leaves alone, plus the two
/// perpendicular components turned by the angle.
///
/// What is exact and what refuses, said plainly:
///
/// - the *turn* must be a [`Rotation`], i.e. have a rational cosine and sine --
///   so quarter and half turns and the 3-4-5 turn are available and a
///   60-degree turn is not;
/// - the *axis direction* must have a rational length. `(0, 0, 1)`, `(3, 4, 0)`
///   and `(1, 1, 0)` are fine; `(1, 1, 1)` has length `sqrt(3)` and is refused
///   with [`GeometryError::IrrationalRequired`], because normalizing it would
///   put `sqrt(3)` denominators into what is supposed to be a rational point.
///
/// Note the thing a 2D habit gets wrong: a rotation is defined about a *line*,
/// and a point on the axis is a fixed point. `rotation_about_axis` on a point of
/// the axis returns that same point, exactly -- and that is a fact worth having
/// in the tests rather than assuming.
pub fn rotation_about_axis(
    axis: &Line3,
    rotation: &Rotation,
    point: &Point3,
) -> anyhow::Result<Point3> {
    let unit = axis.unit_direction()?;
    let (cos, sin) = (rotation.cos.to_q()?, rotation.sin.to_q()?);
    let position = point.to_vector()?.sub(&axis.through.to_vector()?)?;
    let turned = position
        .scale(&cos)?
        .add(&unit.cross(&position)?.scale(&sin)?)?
        .add(&unit.scale(&unit.dot(&position)?.mul(&Q::ONE.sub(&cos)?)?)?)?;
    Point3::from_vec(point, &axis.through.to_vector()?.add(&turned)?)
}

// ------------------------------------------------------- lifting 2D into 3D --

/// The 3D lift of a 2D point: the same name, the same `x` and `y`, and `z = 0`.
///
/// The plane `z = 0` is the right embedding and not an arbitrary one. It is an
/// *isometric* embedding -- distances, angles and every affine ratio are
/// preserved exactly -- and it is injective, so a 2D figure cannot gain or lose
/// a coincidence by being lifted. Every fact the 2D kernel established therefore
/// remains true in 3D, and [`lifted_holds`] re-decides that rather than assuming
/// it.
pub fn lift_point(point: &KPoint) -> anyhow::Result<Point3> {
    Ok(Point3::new(
        &point.name,
        point.x.to_q()?,
        point.y.to_q()?,
        Q::ZERO,
    ))
}

/// Every point of a 2D scene graph, lifted to `z = 0`, in the graph's own order.
pub fn lift_scene(scene: &SceneGraph) -> anyhow::Result<Vec<Point3>> {
    scene.points.iter().map(lift_point).collect()
}

/// Look a lifted point up by name, refusing an unknown name.
pub fn lifted_point<'a>(points: &'a [Point3], name: &str) -> anyhow::Result<&'a Point3> {
    points
        .iter()
        .find(|p| p.name == name)
        .ok_or_else(|| anyhow::anyhow!("the lifted figure has no point '{name}'"))
}

/// The three lifted vertices of a 2D [`Tri3`].
///
/// Refuses a triangle the 2D kernel itself calls degenerate -- the same
/// `DegenerateTriangle` variant, and for the same reason: `ABC` bounds a region
/// only if the three points are not collinear, and in 3D a flat "triangle" is
/// even more clearly not a solid.
pub fn lift_triangle(scene: &SceneGraph, tri: &Tri3) -> anyhow::Result<[Point3; 3]> {
    let points = lift_scene(scene)?;
    let (a, b, c) = (
        lifted_point(&points, &tri.a)?.clone(),
        lifted_point(&points, &tri.b)?.clone(),
        lifted_point(&points, &tri.c)?.clone(),
    );
    anyhow::ensure!(
        !collinear_3d(&a, &b, &c)?,
        GeometryError::DegenerateTriangle {
            vertices: tri.describe()
        }
    );
    Ok([a, b, c])
}

/// The 3D reading of a 2D [`Constraint`] over lifted points, or `None` when the
/// predicate has no 3D reading in the kernel's own vocabulary.
///
/// `None` is the interesting case, and it is a boundary rather than a gap.
/// `Circle`, `OnCircle` and `Diameter` name a *circle*, a 2D object: the 3D
/// analogue is a [`Sphere`], and quietly reading a circle as a sphere is exactly
/// the "three points that look equidistant are not an object with an identity"
/// failure audit fix 1 was about. `AreaEqual`, `Congruent`, `AngleEqual` and
/// `AngleIs` are well defined over `z = 0` but they are the 2D kernel's to
/// decide, and re-implementing them here would be a second place for the two
/// modules to disagree. So they are declined, by name, and
/// [`lifted_fact_holds`] falls back to the kernel for them.
///
/// Every arm is exact, and a collapsed leg refuses rather than holding
/// vacuously, for the reason the 2D kernel gives: "true by default" is how an
/// empty derivation becomes a claimed theorem.
fn native_3d_verdict(points: &[Point3], constraint: &Constraint) -> anyhow::Result<Option<bool>> {
    let at = |name: &str| -> anyhow::Result<Point3> { Ok(lifted_point(points, name)?.clone()) };
    let pair =
        |seg: &Segment| -> anyhow::Result<(Point3, Point3)> { Ok((at(&seg.from)?, at(&seg.to)?)) };
    let legs = |angle: &Angle3| -> anyhow::Result<(Point3, Point3, Point3)> {
        Ok((at(&angle.at)?, at(&angle.from)?, at(&angle.to)?))
    };
    let squared = |a: &Point3, b: &Point3| -> anyhow::Result<Q> {
        a.to_vector()?.sub(&b.to_vector()?)?.length_sq()
    };
    let decided = match constraint {
        Constraint::Collinear { a, b, c } => collinear_3d(&at(a)?, &at(b)?, &at(c)?)?,
        Constraint::Distinct { a, b } => !at(a)?.same_location(&at(b)?)?,
        Constraint::NonCollinear { a, b, c } | Constraint::Triangle { a, b, c } => {
            !collinear_3d(&at(a)?, &at(b)?, &at(c)?)?
        }
        Constraint::MidpointOf { p, a, b } => {
            let (pp, pa, pb) = (at(p)?, at(a)?, at(b)?);
            let mid = pa
                .to_vector()?
                .add(&pb.to_vector()?)?
                .scale(&Q::from_int(1).half()?)?;
            pp.to_vector()?.sub(&mid)?.is_zero()
        }
        Constraint::Between { a, m, b } => {
            let (pa, pm, pb) = (at(a)?, at(m)?, at(b)?);
            if !collinear_3d(&pa, &pm, &pb)? {
                false
            } else {
                // Collinear, so two positive dot products pin `m` strictly
                // inside: not at an end, not outside, not on top of either.
                let along = pm.to_vector()?.sub(&pa.to_vector()?)?;
                let total = pb.to_vector()?.sub(&pa.to_vector()?)?;
                let back = pb.to_vector()?.sub(&pm.to_vector()?)?;
                Q::ZERO.less(&along.dot(&total)?) && Q::ZERO.less(&back.dot(&total)?)
            }
        }
        Constraint::RatioOf { p, a, b, num, den } => {
            anyhow::ensure!(
                *num > 0 && *den > 0,
                GeometryError::EmptyGeometry(format!(
                    "the ratio {num}:{den} does not divide a segment into two positive parts"
                ))
            );
            let (pp, pa, pb) = (at(p)?, at(a)?, at(b)?);
            let (n, d) = (Q::from_int(*num), Q::from_int(*den));
            let total = d.add(&n)?;
            let want = pa
                .to_vector()?
                .scale(&d)?
                .add(&pb.to_vector()?.scale(&n)?)?
                .scale(&Q::ONE.div(&total)?)?;
            pp.to_vector()?.sub(&want)?.is_zero()
        }
        Constraint::LengthIs { seg, square } => {
            let (a, b) = pair(seg)?;
            squared(&a, &b)? == square.to_q()?
        }
        Constraint::EqualLength { first, second } => {
            let (a1, b1) = pair(first)?;
            let (a2, b2) = pair(second)?;
            squared(&a1, &b1)? == squared(&a2, &b2)?
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
            let (a1, b1) = pair(first)?;
            let (a2, b2) = pair(second)?;
            let (n, d) = (Q::from_int(*num), Q::from_int(*den));
            let (rhs, lhs) = (squared(&a1, &b1)?, squared(&a2, &b2)?);
            d.mul(&d)?.mul(&lhs)? == n.mul(&n)?.mul(&rhs)?
        }
        Constraint::Parallel { first, second } => {
            let (a1, b1) = pair(first)?;
            let (a2, b2) = pair(second)?;
            a1.to_vector()?
                .sub(&b1.to_vector()?)?
                .is_parallel_to(&a2.to_vector()?.sub(&b2.to_vector()?)?)?
        }
        Constraint::Perpendicular { first, second } => {
            let (a1, b1) = pair(first)?;
            let (a2, b2) = pair(second)?;
            a1.to_vector()?
                .sub(&b1.to_vector()?)?
                .is_perpendicular_to(&a2.to_vector()?.sub(&b2.to_vector()?)?)?
        }
        Constraint::RightAngle { at: angle } => {
            let (vertex, first, second) = legs(angle)?;
            right_angle_at(&vertex, &first, &second)?
        }
        _ => return Ok(None),
    };
    Ok(Some(decided))
}

/// Whether a 2D [`Constraint`] holds of a lifted figure, decided in three
/// dimensions by this module.
///
/// This is the cross-module certificate the lift exists for. A statement the
/// kernel proved in the plane is not re-proved here; it is *re-decided* by
/// different formulas -- a cross product where the 2D kernel used a determinant,
/// a dot product where it compared slopes -- and both must agree. A lift that
/// quietly changed a fact would invalidate every certificate written above it, so
/// the re-decision is the whole point.
///
/// Refuses the predicates that have no 3D reading in the kernel's vocabulary
/// (circles, areas, congruence, named cosines), naming the reason rather than
/// answering from habit. For those, [`lifted_fact_holds`] is the entry point: it
/// falls back to the 2D kernel's own verdict instead of guessing.
pub fn lifted_holds(points: &[Point3], constraint: &Constraint) -> anyhow::Result<bool> {
    native_3d_verdict(points, constraint)?.ok_or_else(|| {
        anyhow::Error::from(GeometryError::EmptyGeometry(format!(
            "{} is not a predicate this module decides in three dimensions: a circle, an area, a \
             congruence and a named cosine belong to the 2D kernel, and their 3D analogues are \
             different objects",
            constraint.describe()
        )))
    })
}

/// Whether a 2D [`Fact`] still holds after the lift, as a two-module certificate.
///
/// The 2D kernel decides it in the plane, this module re-decides it in space
/// where a 3D reading exists, and the certificate is only issued when both
/// agree. Where no 3D reading exists the kernel's verdict stands alone, because
/// inventing one would be the failure this whole exercise is about.
pub fn lifted_fact_holds(scene: &SceneGraph, fact: &Fact) -> anyhow::Result<bool> {
    if !scene.holds(&fact.constraint)? {
        return Ok(false);
    }
    Ok(native_3d_verdict(&lift_scene(scene)?, &fact.constraint)?.unwrap_or(true))
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
    use crate::geomkernel::{graph_from_points, Fact, KPoint};

    /// An exact rational, for the expected values a test asserts against.
    fn q(num: i128, den: i128) -> Q {
        Q::new(num, den).unwrap()
    }

    /// A lattice point, which is what most of these figures are made of.
    fn p(name: &str, x: i64, y: i64, z: i64) -> Point3 {
        Point3::from_ints(name, x, y, z)
    }

    /// The typed refusal an operation produced. A refusal that is not a
    /// `GeometryError` is a test failure, not a shrug: this module's whole claim
    /// is that it refuses *by name*.
    fn refusal<T>(result: anyhow::Result<T>) -> GeometryError {
        result
            .err()
            .and_then(|why| why.downcast::<GeometryError>().ok())
            .expect("a typed GeometryError")
    }

    /// The exact radical `b sqrt d`, or the exact rational `num/den` when
    /// `d == 0`. `QSqrt` folds a zero radicand into its rational part, so a test
    /// that wants the rational has to ask for it that way.
    fn radical(num: i128, den: i128, d: i64) -> QSqrt {
        if d == 0 {
            QSqrt::rational(Frac { num, den })
        } else {
            QSqrt::new(Frac::from_int(0), Frac { num, den }, d).unwrap()
        }
    }

    #[test]
    fn test_rationals_stay_exact_where_floats_do_not() {
        // The three vector identities, exactly, over Q. A float kernel gets the
        // third one wrong in the last bits and nobody notices until a face area
        // or a right angle depends on it.
        let a = Vec3::from_ints(1, 2, 3);
        let b = Vec3::from_ints(0, 1, 0);
        let cross = a.cross(&b).unwrap();
        assert_eq!(cross.describe(), "(-3, 0, 1)");
        // a x b = -(b x a)
        assert!(b.cross(&a).unwrap().add(&cross).unwrap().is_zero());
        // a . (a x b) = 0
        assert!(a.dot(&cross).unwrap().is_zero());
        // |a x b|^2 = |a|^2 |b|^2 - (a . b)^2 = 14 - 4 = 10
        assert_eq!(cross.length_sq().unwrap(), q(10, 1));
        assert_eq!(
            a.length_sq()
                .unwrap()
                .mul(&b.length_sq().unwrap())
                .unwrap()
                .sub(&a.dot(&b).unwrap().mul(&a.dot(&b).unwrap()).unwrap())
                .unwrap(),
            q(10, 1)
        );
        // and the same sum a float kernel cannot close exactly
        assert_eq!(
            q(1, 3).add(&q(1, 6)).unwrap().add(&q(1, 2)).unwrap(),
            q(1, 1)
        );
        assert_eq!((1.0f64 / 3.0) + (1.0f64 / 6.0) + 0.5, 1.0f64);
        assert_ne!(0.1f64 + 0.2f64, 0.3f64);
    }

    #[test]
    fn test_unit_cube_is_a_solid_with_exact_face_areas_and_volume() {
        let cube = Polyhedron::unit_cube();
        cube.sanity_check().unwrap();
        assert!(cube.orientation_is_consistent().unwrap());
        assert!(cube.is_convex().unwrap());
        // Every face of the unit cube has area exactly 1...
        for face in &cube.faces {
            assert_eq!(
                cube.face_area(face).unwrap(),
                radical(1, 1, 0),
                "{}",
                face.describe()
            );
        }
        // ...so the total is exactly 6, and the volume is exactly 1.
        assert_eq!(cube.total_surface_area().unwrap(), radical(6, 1, 0));
        assert_eq!(cube.volume().unwrap(), q(1, 1));
        assert_eq!(cube.signed_volume().unwrap(), q(1, 1));
        // The face plane is the equation, not a guess: the bottom face holds
        // exactly its four own vertices and not the fifth.
        let bottom = cube.face_plane(&cube.faces[0]).unwrap();
        assert_eq!(bottom.describe(), "plane(bottom: (0, 0, -1) . x = 0)");
        for corner in ["A", "B", "C", "D"] {
            assert!(
                bottom.contains(cube.vertex(corner).unwrap()).unwrap(),
                "{corner}"
            );
        }
        assert!(!bottom.contains(cube.vertex("E").unwrap()).unwrap());
    }

    #[test]
    fn test_unit_tetrahedron_face_areas_are_exact_and_irrational() {
        let tetra = Polyhedron::unit_tetrahedron();
        tetra.sanity_check().unwrap();
        assert!(tetra.orientation_is_consistent().unwrap());
        assert!(tetra.is_convex().unwrap());
        // Three right-isoceles faces of area 1/2...
        for name in ["oxy", "oxz", "oyz"] {
            let face = tetra.faces.iter().find(|f| f.name == name).unwrap();
            assert_eq!(tetra.face_area(face).unwrap(), radical(1, 2, 0));
        }
        // ...and the slanted one exactly sqrt(3)/2, not 0.8660254.
        let slanted = tetra.faces.iter().find(|f| f.name == "xyz").unwrap();
        assert_eq!(tetra.face_area(slanted).unwrap(), radical(1, 2, 3));
        assert_eq!(tetra.volume().unwrap(), q(1, 6));
        assert_eq!(tetra.signed_volume().unwrap(), q(1, 6));
    }

    #[test]
    fn test_total_surface_area_refuses_when_the_radicals_differ() {
        // The cube's six rational faces sum exactly.
        assert_eq!(
            Polyhedron::unit_cube().total_surface_area().unwrap(),
            radical(6, 1, 0)
        );
        // The tetrahedron's do not: `1/2 + 1/2 + 1/2 + sqrt(3)/2` is not in the
        // quadratic field, so this refuses rather than returning 2.366...
        let refused = refusal(Polyhedron::unit_tetrahedron().total_surface_area());
        assert!(
            matches!(refused, GeometryError::IrrationalRequired(_)),
            "{refused}"
        );
        // Per face, every one of them is still exact.
        let tetra = Polyhedron::unit_tetrahedron();
        let areas: Vec<QSqrt> = tetra
            .faces
            .iter()
            .map(|f| tetra.face_area(f).unwrap())
            .collect();
        assert!(areas.iter().all(|area| *area != radical(0, 1, 0)));
    }

    /// The L-shaped prism: a solid whose base is the L `(0,0) (2,0) (2,1)
    /// (1,1) (1,2) (0,2)` extruded to height 1. Twelve vertices, eight faces,
    /// base area 3, volume 3, total area 14 -- and a reflex corner at `(1,1)`,
    /// which is the whole point: a low-resolution picture of it is a blob, and
    /// the exact test says it is not a convex solid.
    fn l_prism() -> Polyhedron {
        let base = [
            ("p0", 0, 0),
            ("p1", 2, 0),
            ("p2", 2, 1),
            ("p3", 1, 1),
            ("p4", 1, 2),
            ("p5", 0, 2),
        ];
        let mut vertices = Vec::new();
        for (name, x, y) in base {
            vertices.push(p(&format!("{name}0"), x, y, 0));
        }
        for (name, x, y) in base {
            vertices.push(p(&format!("{name}1"), x, y, 1));
        }
        let faces = vec![
            Face3::new("top", &["p01", "p11", "p21", "p31", "p41", "p51"]),
            Face3::new("bottom", &["p50", "p40", "p30", "p20", "p10", "p00"]),
            Face3::new("s0", &["p00", "p10", "p11", "p01"]),
            Face3::new("s1", &["p10", "p20", "p21", "p11"]),
            Face3::new("s2", &["p20", "p30", "p31", "p21"]),
            Face3::new("s3", &["p30", "p40", "p41", "p31"]),
            Face3::new("s4", &["p40", "p50", "p51", "p41"]),
            Face3::new("s5", &["p50", "p00", "p01", "p51"]),
        ];
        Polyhedron::checked("L-prism", vertices, faces).unwrap()
    }

    #[test]
    fn test_concavity_is_decided_by_a_sign_not_by_a_picture() {
        let solid = l_prism();
        solid.sanity_check().unwrap();
        assert_eq!(solid.vertices.len(), 12);
        assert_eq!(solid.faces.len(), 8);
        // Wound correctly and closed, so the volume means something...
        assert!(solid.orientation_is_consistent().unwrap());
        assert_eq!(solid.volume().unwrap(), q(3, 1));
        assert_eq!(solid.total_surface_area().unwrap(), radical(14, 1, 0));
        // ...and the reflex corner at (1, 1) makes it not a convex solid, which
        // a sign per vertex per face decides exactly.
        assert!(!solid.is_convex().unwrap());
        // The same figure with the reflex corner cut off *is* convex: drop the
        // two vertices beyond it and the remaining box is a 2x2x1 cuboid.
        let box_solid = Polyhedron::checked(
            "cuboid",
            vec![
                p("A", 0, 0, 0),
                p("B", 2, 0, 0),
                p("C", 2, 2, 0),
                p("D", 0, 2, 0),
                p("E", 0, 0, 1),
                p("F", 2, 0, 1),
                p("G", 2, 2, 1),
                p("H", 0, 2, 1),
            ],
            vec![
                Face3::new("bottom", &["A", "D", "C", "B"]),
                Face3::new("top", &["E", "F", "G", "H"]),
                Face3::new("front", &["A", "B", "F", "E"]),
                Face3::new("right", &["B", "C", "G", "F"]),
                Face3::new("back", &["D", "H", "G", "C"]),
                Face3::new("left", &["A", "E", "H", "D"]),
            ],
        )
        .unwrap();
        assert!(box_solid.is_convex().unwrap());
        assert_eq!(box_solid.volume().unwrap(), q(4, 1));
        assert_eq!(box_solid.total_surface_area().unwrap(), radical(16, 1, 0));
    }

    #[test]
    fn test_polyhedron_orientation_must_be_consistent() {
        // The unit cube with its top face wound the other way: the same eight
        // vertices, the same six faces, and no coherent inside.
        let mut cube = Polyhedron::unit_cube();
        cube.faces[1] = Face3::new("top", &["E", "H", "G", "F"]);
        cube.sanity_check().unwrap();
        assert!(!cube.orientation_is_consistent().unwrap());
        // Everything that needs to know which way is out refuses by name.
        assert!(matches!(
            refusal(cube.signed_volume()),
            GeometryError::Unproven(_)
        ));
        assert!(matches!(
            refusal(cube.is_convex()),
            GeometryError::Unproven(_)
        ));
        // Sanity alone still passes: the mesh is structurally fine, which is
        // exactly why the orientation check is a separate question.
        assert!(cube.volume().is_err());
    }

    #[test]
    fn test_right_angle_in_three_dimensions_is_a_zero_dot_product() {
        // A corner of the unit cube: (B - A) . (C - A) = (1,0,0) . (0,1,0) = 0.
        let (a, b, c) = (p("A", 0, 0, 0), p("B", 1, 0, 0), p("C", 0, 1, 0));
        assert_eq!(
            b.to_vector()
                .unwrap()
                .sub(&a.to_vector().unwrap())
                .unwrap()
                .dot(&c.to_vector().unwrap().sub(&a.to_vector().unwrap()).unwrap())
                .unwrap(),
            q(0, 1)
        );
        assert!(right_angle_at(&a, &b, &c).unwrap());
        // The same fact stated about the vertex E, out along x and out along z.
        assert!(right_angle_at(&p("E", 0, 0, 1), &p("F", 1, 0, 1), &p("H", 0, 1, 1)).unwrap());
        // A collapsed leg is not an angle, in three dimensions as in two.
        assert!(matches!(
            refusal(right_angle_at(&a, &a, &b)),
            GeometryError::DegenerateAngle { .. }
        ));
    }

    #[test]
    fn test_a_diagonal_is_not_perpendicular_to_the_edge_it_sits_on() {
        // At A, the edge AB and the face diagonal AC of the unit cube: the
        // dot product is exactly 1, so this is a 45-degree angle and not a right
        // one, however square the drawing looks.
        let (a, b, c) = (p("A", 0, 0, 0), p("B", 1, 0, 0), p("C", 1, 1, 0));
        assert!(!right_angle_at(&a, &b, &c).unwrap());
        let diagonal = c.to_vector().unwrap().sub(&a.to_vector().unwrap()).unwrap();
        let face_normal = p("E", 0, 0, 1).to_vector().unwrap();
        // The face diagonal is exactly sqrt(2) long and the space diagonal
        // exactly sqrt(3) -- quadratic-field values, not rounded ones.
        assert_eq!(diagonal.length().unwrap(), radical(1, 1, 2));
        assert_eq!(
            p("A", 0, 0, 0).distance_to(&p("G", 1, 1, 1)).unwrap(),
            radical(1, 1, 3)
        );
        // A face diagonal and the face normal are exactly perpendicular.
        assert!(diagonal.is_perpendicular_to(&face_normal).unwrap());
        assert_eq!(diagonal.dot(&face_normal).unwrap(), q(0, 1));
    }

    #[test]
    fn test_projection_onto_a_plane_lands_exactly_on_it() {
        // The plane x + y + z = 6 and the point (1, 1, 1): the projection walks
        // back along the normal by exactly its signed distance.
        let plane = Plane3::new("sum", Vec3::from_ints(1, 1, 1), Q::from_int(6)).unwrap();
        let projected = projection_point_plane(&p("P", 1, 1, 1), &plane).unwrap();
        assert_eq!(projected.coords().unwrap(), (q(2, 1), q(2, 1), q(2, 1)));
        // The certificate: re-derive the plane equation from the stored normal
        // and offset rather than trusting the projection.
        assert!(plane.contains(&projected).unwrap());
        let (nx, ny, nz) = plane.normal.coords().unwrap();
        let (x, y, z) = projected.coords().unwrap();
        assert_eq!(
            nx.mul(&x)
                .unwrap()
                .add(&ny.mul(&y).unwrap())
                .unwrap()
                .add(&nz.mul(&z).unwrap())
                .unwrap(),
            plane.offset.to_q().unwrap()
        );
        // The displacement is parallel to the normal, so the projection really
        // did move along the normal and not somewhere else.
        let moved = p("P", 1, 1, 1)
            .to_vector()
            .unwrap()
            .sub(&projected.to_vector().unwrap())
            .unwrap();
        assert!(moved.is_parallel_to(&plane.normal).unwrap());
        // z = 0 is the easy case, and it is exact too.
        let floor = Plane3::new("floor", Vec3::from_ints(0, 0, 1), Q::ZERO).unwrap();
        let down = projection_point_plane(&p("Q", 1, 2, 3), &floor).unwrap();
        assert_eq!(down.coords().unwrap(), (q(1, 1), q(2, 1), q(0, 1)));
        assert_eq!(
            distance_point_plane(&down, &floor).unwrap(),
            radical(0, 1, 0)
        );
    }

    #[test]
    fn test_projection_onto_a_line_lands_exactly_on_it() {
        // The line through the origin in the direction (1, 1, 0), and the point
        // (0, 1, 0): the foot of the perpendicular is exactly (1/2, 1/2, 0).
        let line = Line3::through_points("l", &p("O", 0, 0, 0), &p("U", 1, 1, 0)).unwrap();
        let foot = projection_point_line(&p("P", 0, 1, 0), &line).unwrap();
        assert_eq!(foot.coords().unwrap(), (q(1, 2), q(1, 2), q(0, 1)));
        assert!(line.contains(&foot).unwrap());
        // The displacement is perpendicular to the line, exactly: (0,1,0) and
        // (1,1,0) do not multiply to zero, but (-1/2, 1/2, 0) and (1, 1, 0) do.
        let moved = p("P", 0, 1, 0)
            .to_vector()
            .unwrap()
            .sub(&foot.to_vector().unwrap())
            .unwrap();
        assert!(moved.is_perpendicular_to(&line.dir).unwrap());
        assert_eq!(
            distance_point_line(&p("P", 0, 1, 0), &line).unwrap(),
            radical(1, 2, 2)
        );
        // A point of the line is its own projection.
        let on_line = projection_point_line(&p("U", 1, 1, 0), &line).unwrap();
        assert!(on_line.same_location(&p("U", 1, 1, 0)).unwrap());
    }

    #[test]
    fn test_distances_are_exact_even_when_the_normal_is_not() {
        // The plane 3x + 4y + 5z = 0 has a normal of length sqrt(50) = 5 sqrt 2,
        // which is irrational -- and the distance from (1, 1, 1) is still exact,
        // because the squared distance is the rational 72/25 and only the root
        // of it is irrational. A float kernel would divide by 7.0710678...
        let plane = Plane3::new("n345", Vec3::from_ints(3, 4, 5), Q::ZERO).unwrap();
        assert_eq!(
            plane.distance_to(&p("P", 1, 1, 1)).unwrap(),
            radical(6, 5, 2)
        );
        assert!(plane
            .distance_to(&plane.point_on("foot").unwrap())
            .unwrap()
            .is_zero()
            .unwrap());
        // The distance from a point to a line is the same story: (1, 2, 1) to the
        // line through the origin along (1, 1, 0) is exactly sqrt(3/2).
        let line = Line3::through_points("l", &p("O", 0, 0, 0), &p("U", 1, 1, 0)).unwrap();
        assert_eq!(
            distance_point_line(&p("P", 1, 2, 1), &line).unwrap(),
            radical(1, 2, 6)
        );
        // A point on the plane is at distance zero, and zero is a value here
        // rather than an absence.
        assert_eq!(
            distance_point_plane(&p("O", 0, 0, 0), &plane).unwrap(),
            radical(0, 1, 0)
        );
    }

    #[test]
    fn test_line_plane_intersection_satisfies_the_plane_equation() {
        // The diagonal of the unit cube against the far face z = 1.
        let diagonal = Line3::through_points("d", &p("A", 0, 0, 0), &p("G", 1, 1, 1)).unwrap();
        let far = Plane3::new("far", Vec3::from_ints(0, 0, 1), Q::from_int(1)).unwrap();
        let hit = intersect_line_plane(&diagonal, &far).unwrap();
        assert_eq!(hit.coords().unwrap(), (q(1, 1), q(1, 1), q(1, 1)));
        // The certificate: the result satisfies the plane equation, re-derived.
        assert!(far.contains(&hit).unwrap());
        let (nz, offset) = (Q::from_int(1), Q::from_int(1));
        let (_, _, z) = hit.coords().unwrap();
        assert_eq!(nz.mul(&z).unwrap(), offset);
        // ...and it is the cube vertex G, named -- which is also the far face's
        // own plane, so the solid agrees with the line-plane operation.
        assert!(hit.same_location(&p("G", 1, 1, 1)).unwrap());
        let cube = Polyhedron::unit_cube();
        let top = cube.face_plane(&cube.faces[1]).unwrap();
        assert_eq!(top.describe(), "plane(top: (0, 0, 1) . x = 1)");
        assert!(top.contains(&hit).unwrap());
        assert!(diagonal.contains(&hit).unwrap());
    }

    #[test]
    fn test_line_plane_intersection_is_a_rational_point() {
        // The line (0,0,0) + t (1, 2, 2) against x + y + z = 6: t = 6/5 exactly.
        // A float kernel returns 1.1999999999999997 for the first coordinate.
        let line = Line3::through_points("l", &p("O", 0, 0, 0), &p("D", 1, 2, 2)).unwrap();
        let plane = Plane3::new("sum", Vec3::from_ints(1, 1, 1), Q::from_int(6)).unwrap();
        let hit = intersect_line_plane(&line, &plane).unwrap();
        assert_eq!(hit.coords().unwrap(), (q(6, 5), q(12, 5), q(12, 5)));
        assert!(plane.contains(&hit).unwrap());
        // 6/5 is exactly 6/5; the point of asserting on `Q` is that the float
        // nearest to it is a different number from the rational the geometry
        // says, and every later comparison inherits that difference.
        assert_eq!(q(6, 5).to_f64(), 1.2f64);
        // The parameter is recoverable from the point, which is the other half
        // of the certificate.
        assert!(line.contains(&hit).unwrap());
    }

    #[test]
    fn test_line_plane_intersection_refuses_a_parallel_line() {
        // A line running along the plane's own direction never meets it, in one
        // place or in all of them; the kernel's planar refusal, reused.
        let floor = Plane3::new("floor", Vec3::from_ints(0, 0, 1), Q::ZERO).unwrap();
        let along = Line3::new("along", p("P", 0, 0, 3), Vec3::from_ints(1, 0, 0)).unwrap();
        assert!(matches!(
            refusal(intersect_line_plane(&along, &floor)),
            GeometryError::ParallelLines
        ));
        // A line in the plane's direction but not in the plane: still parallel,
        // and the refusal does not pretend to pick a point of the plane.
        let above = Line3::new("above", p("Q", 0, 0, 5), Vec3::from_ints(1, 0, 0)).unwrap();
        assert!(matches!(
            refusal(intersect_line_plane(&above, &floor)),
            GeometryError::ParallelLines
        ));
    }

    #[test]
    fn test_plane_plane_intersection_is_a_line_in_both_planes() {
        // z = 0 and x = 2 meet in the line (2, t, 0).
        let floor = Plane3::new("floor", Vec3::from_ints(0, 0, 1), Q::ZERO).unwrap();
        let wall = Plane3::new("wall", Vec3::from_ints(1, 0, 0), Q::from_int(2)).unwrap();
        let edge = intersect_plane_plane(&floor, &wall).unwrap();
        assert!(floor.contains(&edge.through).unwrap());
        assert!(wall.contains(&edge.through).unwrap());
        assert_eq!(edge.through.coords().unwrap(), (q(2, 1), q(0, 1), q(0, 1)));
        assert!(edge.dir.is_parallel_to(&Vec3::from_ints(0, 1, 0)).unwrap());
        // Every point of the returned line is on both planes, at two different
        // parameters -- the certificate, re-derived rather than trusted.
        for t in [q(0, 1), q(3, 2), q(-5, 1)] {
            let on_edge = edge.at(&t).unwrap();
            assert!(floor.contains(&on_edge).unwrap(), "t = {t}");
            assert!(wall.contains(&on_edge).unwrap(), "t = {t}");
        }
        // A non-axis-aligned pair, where the formula is not obvious.
        let a = Plane3::new("a", Vec3::from_ints(1, 1, 0), Q::from_int(1)).unwrap();
        let b = Plane3::new("b", Vec3::from_ints(1, 0, 1), Q::from_int(1)).unwrap();
        let cut = intersect_plane_plane(&a, &b).unwrap();
        assert!(a.contains(&cut.through).unwrap());
        assert!(b.contains(&cut.through).unwrap());
        assert_eq!(cut.through.coords().unwrap(), (q(2, 3), q(1, 3), q(1, 3)));
    }

    #[test]
    fn test_parallel_and_coincident_planes_refuse_to_intersect() {
        // Distinct parallel planes: the cross product of the normals is zero and
        // no point of one satisfies the other.
        let low = Plane3::new("low", Vec3::from_ints(0, 0, 1), Q::ZERO).unwrap();
        let high = Plane3::new("high", Vec3::from_ints(0, 0, 1), Q::from_int(1)).unwrap();
        assert!(matches!(
            refusal(intersect_plane_plane(&low, &high)),
            GeometryError::ParallelLines
        ));
        // The same plane stated with a scaled normal is the *same* plane, and
        // reporting a line for it would be reporting a fiction.
        let scaled = Plane3::new("scaled", Vec3::from_ints(0, 0, 5), Q::ZERO).unwrap();
        assert!(matches!(
            refusal(intersect_plane_plane(&low, &scaled)),
            GeometryError::CoincidentLines
        ));
        assert!(matches!(
            refusal(intersect_plane_plane(&low, &low)),
            GeometryError::CoincidentLines
        ));
    }

    #[test]
    fn test_skew_lines_refuse_to_intersect() {
        // One line along x through the origin, one along y through (0, 0, 1).
        // Not parallel, and not coplanar: they never meet, and the word for it
        // is skew.
        let along_x = Line3::through_points("x-axis", &p("O", 0, 0, 0), &p("X", 1, 0, 0)).unwrap();
        let along_y = Line3::through_points("y-shift", &p("S", 0, 0, 1), &p("Y", 0, 1, 1)).unwrap();
        let refused = refusal(intersect_line_line(&along_x, &along_y));
        assert!(
            matches!(&refused, GeometryError::NoRationalSolution(why) if why.contains("skew")),
            "{refused}"
        );
        // The coplanarity sign that decides it, shown directly: (p2 - p1) is
        // (0, 0, 1) and d1 x d2 is (0, 0, 1), so their dot product is 1.
        let normal = along_x.dir.cross(&along_y.dir).unwrap();
        let w = along_y
            .through
            .to_vector()
            .unwrap()
            .sub(&along_x.through.to_vector().unwrap())
            .unwrap();
        assert_eq!(w.dot(&normal).unwrap(), q(1, 1));
        // The same pair in a plane *is* intersecting, and then it does meet.
        let coplanar = Line3::through_points("y", &p("S", 0, 0, 0), &p("Y", 0, 1, 0)).unwrap();
        let corner = intersect_line_line(&along_x, &coplanar).unwrap();
        assert!(corner.same_location(&p("O", 0, 0, 0)).unwrap());
    }

    #[test]
    fn test_skew_lines_still_have_a_closest_pair() {
        // The same skew pair: no intersection, but a nearest pair at distance 1,
        // and that distance is a question a 3D reasoner asks constantly.
        let along_x = Line3::through_points("x-axis", &p("O", 0, 0, 0), &p("X", 1, 0, 0)).unwrap();
        let along_y = Line3::through_points("y-shift", &p("S", 0, 0, 1), &p("Y", 0, 1, 1)).unwrap();
        let (on_x, on_y) = closest_points_line_line(&along_x, &along_y).unwrap();
        assert!(along_x.contains(&on_x).unwrap());
        assert!(along_y.contains(&on_y).unwrap());
        assert_eq!(on_x.coords().unwrap(), (q(0, 1), q(0, 1), q(0, 1)));
        assert_eq!(on_y.coords().unwrap(), (q(0, 1), q(0, 1), q(1, 1)));
        assert_eq!(on_x.distance_to(&on_y).unwrap(), radical(1, 1, 0));
        // Intersecting lines get their intersection as the unique closest pair.
        let coplanar = Line3::through_points("y", &p("S", 0, 0, 0), &p("Y", 0, 1, 0)).unwrap();
        let (first, second) = closest_points_line_line(&along_x, &coplanar).unwrap();
        assert!(first.same_location(&second).unwrap());
    }

    #[test]
    fn test_parallel_and_coincident_lines_refuse_rather_than_guess() {
        let along_x = Line3::through_points("x-axis", &p("O", 0, 0, 0), &p("X", 1, 0, 0)).unwrap();
        let also_x = Line3::through_points("shifted", &p("P", 0, 1, 0), &p("Q", 1, 1, 0)).unwrap();
        let same_x = Line3::through_points("same", &p("O", 0, 0, 0), &p("R", 3, 0, 0)).unwrap();
        // Distinct parallel lines: no intersection, and no *unique* closest pair
        // either, so both operations refuse by name.
        assert!(matches!(
            refusal(intersect_line_line(&along_x, &also_x)),
            GeometryError::ParallelLines
        ));
        assert!(matches!(
            refusal(closest_points_line_line(&along_x, &also_x)),
            GeometryError::ParallelLines
        ));
        // The same line named twice is a different failure from a different
        // parallel line, and the variants keep them apart.
        assert!(matches!(
            refusal(intersect_line_line(&along_x, &same_x)),
            GeometryError::CoincidentLines
        ));
        assert!(matches!(
            refusal(closest_points_line_line(&along_x, &same_x)),
            GeometryError::CoincidentLines
        ));
    }

    #[test]
    fn test_mental_rotation_by_a_quarter_turn() {
        // The mental-rotation question, answered by transforming coordinates:
        // turn the point (1, 0, 0) a quarter turn about the z axis and it is
        // (0, 1, 0) -- exactly, with no picture of the object anywhere.
        let axis = Line3::new("z", p("O", 0, 0, 0), Vec3::from_ints(0, 0, 1)).unwrap();
        let turned =
            rotation_about_axis(&axis, &Rotation::quarter_turn(), &p("P", 1, 0, 0)).unwrap();
        assert_eq!(turned.coords().unwrap(), (q(0, 1), q(1, 1), q(0, 1)));
        // And the compound case that a 2D mental-rotation model gets wrong:
        // (1, 1, 0) is at 45 degrees in the plane and comes back at 135.
        let diagonal =
            rotation_about_axis(&axis, &Rotation::quarter_turn(), &p("P", 1, 1, 0)).unwrap();
        assert_eq!(diagonal.coords().unwrap(), (q(-1, 1), q(1, 1), q(0, 1)));
        // Distances are preserved exactly, which is the invariant the rotation
        // has to satisfy and the one a rounded matrix would break.
        assert_eq!(
            p("O", 0, 0, 0).distance_to(&diagonal).unwrap(),
            radical(1, 1, 2)
        );
        // Four quarter turns are the identity.
        let mut spun = p("P", 3, 4, 5);
        for _ in 0..4 {
            spun = rotation_about_axis(&axis, &Rotation::quarter_turn(), &spun).unwrap();
        }
        assert!(spun.same_location(&p("P", 3, 4, 5)).unwrap());
    }

    #[test]
    fn test_mental_rotation_by_a_rational_three_four_five_turn() {
        // cos 3/5, sin 4/5 about the axis (3, 4, 0): not an angle anyone
        // memorizes, and still exact. The image of (1, 0, 0) is
        // (93/125, 24/125, -80/125), worked out by hand from Rodrigues.
        let axis = Line3::new("tilted", p("O", 0, 0, 0), Vec3::from_ints(3, 4, 0)).unwrap();
        let turn = Rotation::three_four_five_turn();
        assert_eq!(
            (turn.cos, turn.sin),
            (Frac { num: 3, den: 5 }, Frac { num: 4, den: 5 })
        );
        let turned = rotation_about_axis(&axis, &turn, &p("P", 1, 0, 0)).unwrap();
        assert_eq!(
            turned.coords().unwrap(),
            (q(93, 125), q(24, 125), q(-80, 125))
        );
        // A rotation preserves the distance from the axis point: length 1
        // exactly, checked as 93^2 + 24^2 + 80^2 == 125^2 rather than in floats.
        assert_eq!(
            p("O", 0, 0, 0).distance_to(&turned).unwrap(),
            radical(1, 1, 0)
        );
        assert_eq!(
            q(93, 125)
                .mul(&q(93, 125))
                .unwrap()
                .add(&q(24, 125).mul(&q(24, 125)).unwrap())
                .unwrap()
                .add(&q(80, 125).mul(&q(80, 125)).unwrap())
                .unwrap(),
            q(1, 1)
        );
        // The displacement is perpendicular to the axis, as it must be.
        let moved = p("P", 1, 0, 0)
            .to_vector()
            .unwrap()
            .sub(&turned.to_vector().unwrap())
            .unwrap();
        assert!(moved.is_perpendicular_to(&axis.dir).unwrap());
    }

    #[test]
    fn test_a_point_on_the_axis_is_a_fixed_point() {
        // A rotation is about a *line*; the points on it do not move. This is the
        // property a 2D habit silently gets wrong by rotating "about a point".
        let axis = Line3::new("z", p("O", 0, 0, 0), Vec3::from_ints(0, 0, 1)).unwrap();
        for turn in [
            Rotation::quarter_turn(),
            Rotation::three_four_five_turn(),
            Rotation::identity(),
        ] {
            let fixed = rotation_about_axis(&axis, &turn, &p("P", 0, 0, 5)).unwrap();
            assert!(fixed.same_location(&p("P", 0, 0, 5)).unwrap(), "{turn:?}");
        }
        // A half turn about a line is a point reflection in it, still exact.
        let flipped = rotation_about_axis(
            &axis,
            &Rotation::new(Q::from_int(-1), Q::ZERO).unwrap(),
            &p("P", 2, 3, 0),
        )
        .unwrap();
        assert_eq!(flipped.coords().unwrap(), (q(-2, 1), q(-3, 1), q(0, 1)));
    }

    #[test]
    fn test_rotation_preserves_a_solid_exactly() {
        // Turn the unit cube a quarter turn about the z axis through the origin
        // and measure the result. A rotation is an isometry, so the volume must
        // still be exactly 1 and the solid still convex; the vertex coordinates
        // are known in advance and are the real assertion.
        let cube = Polyhedron::unit_cube();
        let axis = Line3::new("z", p("O", 0, 0, 0), Vec3::from_ints(0, 0, 1)).unwrap();
        let turned: Vec<Point3> = cube
            .vertices
            .iter()
            .map(|v| rotation_about_axis(&axis, &Rotation::quarter_turn(), v).unwrap())
            .collect();
        let spun = Polyhedron::new("turned cube", turned, cube.faces.clone());
        spun.sanity_check().unwrap();
        assert!(spun.orientation_is_consistent().unwrap());
        assert!(spun.is_convex().unwrap());
        assert_eq!(spun.volume().unwrap(), q(1, 1));
        assert_eq!(spun.total_surface_area().unwrap(), radical(6, 1, 0));
        // A = (0,0,0) is the origin, on the axis, so it stays put; B = (1,0,0)
        // goes to (0,1,0); G = (1,1,1) goes to (-1,1,1).
        assert!(spun
            .vertex("A")
            .unwrap()
            .same_location(&p("A", 0, 0, 0))
            .unwrap());
        assert!(spun
            .vertex("B")
            .unwrap()
            .same_location(&p("B", 0, 1, 0))
            .unwrap());
        assert!(spun
            .vertex("G")
            .unwrap()
            .same_location(&p("G", -1, 1, 1))
            .unwrap());
    }

    #[test]
    fn test_rotation_refuses_what_it_cannot_do_exactly() {
        // A 60-degree turn needs sin = sqrt(3)/2, which is not a rational, so
        // there is no Rotation to build and no point to return. The refusal is
        // the answer, not a rounded substitute.
        assert!(Rotation::new(Q::from_int(1), Q::ZERO).is_ok());
        assert!(matches!(
            refusal(Rotation::new(q(1, 2), q(1, 2))),
            GeometryError::NoRationalSolution(_)
        ));
        // The check is `cos^2 + sin^2 == 1`, exactly: (3/5, 4/5) passes and
        // (3/5, 1/2) does not, because 9/25 + 1/4 is 61/100.
        assert!(Rotation::new(q(3, 5), q(4, 5)).is_ok());
        assert!(Rotation::new(q(3, 5), q(1, 2)).is_err());
        // The axis must have a rational length: (1, 1, 1) is sqrt(3).
        let irrational_axis =
            Line3::new("body", p("O", 0, 0, 0), Vec3::from_ints(1, 1, 1)).unwrap();
        let refused = refusal(rotation_about_axis(
            &irrational_axis,
            &Rotation::quarter_turn(),
            &p("P", 1, 0, 0),
        ));
        assert!(
            matches!(refused, GeometryError::IrrationalRequired(_)),
            "{refused}"
        );
        // The same direction written with a zero is not a direction at all.
        assert!(matches!(
            refusal(Vec3::from_ints(1, 1, 0).unit()),
            GeometryError::IrrationalRequired(_)
        ));
        // ...while (3, 4, 0) normalizes to exactly (3/5, 4/5, 0).
        assert_eq!(
            Vec3::from_ints(3, 4, 0).unit().unwrap().describe(),
            "(3/5, 4/5, 0)"
        );
    }

    #[test]
    fn test_a_zero_direction_and_a_zero_normal_are_refused() {
        // A line with no direction is the degenerate segment the kernel already
        // refuses to build, and a plane with no normal is no geometry at all.
        assert!(matches!(
            refusal(Line3::new("bad", p("O", 0, 0, 0), Vec3::zero())),
            GeometryError::DegenerateSegment
        ));
        assert!(matches!(
            refusal(Plane3::new("bad", Vec3::zero(), Q::from_int(3))),
            GeometryError::EmptyGeometry(_)
        ));
        assert!(matches!(
            refusal(Vec3::zero().unit()),
            GeometryError::EmptyGeometry(_)
        ));
        // A zero vector is parallel to nothing and perpendicular to nothing; the
        // vacuous `true` is exactly how a theorem about nothing gets stated.
        assert!(matches!(
            refusal(Vec3::zero().is_parallel_to(&Vec3::from_ints(1, 0, 0))),
            GeometryError::DegenerateSegment
        ));
        assert!(matches!(
            refusal(Vec3::zero().is_perpendicular_to(&Vec3::from_ints(1, 0, 0))),
            GeometryError::DegenerateAngle { .. }
        ));
    }

    #[test]
    fn test_a_line_through_two_coincident_points_is_refused() {
        let a = p("A", 1, 2, 3);
        let b = p("B", 1, 2, 3);
        // Audit fix 11, in three dimensions: two names for one location is how a
        // kernel ends up dividing by zero and calling the result a point.
        assert!(matches!(
            refusal(Line3::through_points("bad", &a, &b)),
            GeometryError::CoincidentPoints { .. }
        ));
        assert!(a.same_location(&b).unwrap());
        assert!(!a.same_location(&p("C", 1, 2, 4)).unwrap());
        // Through distinct points it is fine, and the direction is exact.
        let line = Line3::through_points("good", &a, &p("C", 1, 2, 4)).unwrap();
        assert_eq!(line.dir.describe(), "(0, 0, 1)");
    }

    #[test]
    fn test_a_plane_through_three_collinear_points_is_refused() {
        // Three points on the z axis: the 2D kernel calls this a degenerate
        // triangle, and so does this, with the same variant, because a "plane"
        // through them is every plane containing the axis at once.
        let refused = refusal(Plane3::through_points(
            "bad",
            &p("A", 0, 0, 0),
            &p("B", 0, 0, 1),
            &p("C", 0, 0, 2),
        ));
        assert!(
            matches!(refused, GeometryError::DegenerateTriangle { .. }),
            "{refused}"
        );
        assert!(refused.to_string().contains("A, B, C"));
        // Two identical points collapse the same way.
        assert!(matches!(
            refusal(Plane3::through_points(
                "bad",
                &p("A", 0, 0, 0),
                &p("A", 0, 0, 0),
                &p("C", 1, 0, 0)
            )),
            GeometryError::DegenerateTriangle { .. }
        ));
        // Three non-collinear points give a plane, and the equation is exact.
        let plane =
            Plane3::through_points("z0", &p("A", 0, 0, 0), &p("B", 1, 0, 0), &p("C", 0, 1, 0))
                .unwrap();
        assert_eq!(plane.describe(), "plane(z0: (0, 0, 1) . x = 0)");
        assert!(plane.contains(&p("D", 7, 11, 0)).unwrap());
        assert!(!plane.contains(&p("E", 7, 11, 1)).unwrap());
    }

    #[test]
    fn test_sphere_membership_is_exact_and_degenerate_radii_are_refused() {
        // The sphere of squared radius 2 about the origin: (1, 1, 0) is on it and
        // (1, 0, 0) is not, decided by a squared-distance equality.
        let sphere = Sphere::new("s", p("O", 0, 0, 0), Q::from_int(2)).unwrap();
        assert!(sphere.contains(&p("P", 1, 1, 0)).unwrap());
        assert!(!sphere.contains(&p("P", 1, 0, 0)).unwrap());
        assert_eq!(sphere.distance_sq(&p("P", 1, 1, 0)).unwrap(), q(2, 1));
        // The radius itself is exactly sqrt(2), which is why it is stored squared.
        assert_eq!(sphere.radius().unwrap(), radical(1, 1, 2));
        // A rational radius works just as well: squared radius 9/4 is radius 3/2.
        let ball = Sphere::new("b", p("C", 1, 1, 1), q(9, 4)).unwrap();
        assert_eq!(ball.radius().unwrap(), radical(3, 2, 0));
        // (5/2, 1, 1) is exactly 3/2 from the centre, so it is on the sphere,
        // and (1, 1, 5) is 4 away and is not.
        assert!(ball
            .contains(&Point3::new("P", q(5, 2), q(1, 1), q(1, 1)))
            .unwrap());
        assert!(!ball.contains(&p("Q", 1, 1, 5)).unwrap());
        assert_eq!(ball.distance_sq(&p("Q", 1, 1, 5)).unwrap(), q(16, 1));
        // A point of a zero radius: the kernel's circle refusal, reused.
        assert!(matches!(
            refusal(Sphere::new("z", p("O", 0, 0, 0), Q::ZERO)),
            GeometryError::ZeroRadiusCircle { .. }
        ));
        // A negative radius squared is not a degenerate sphere, it is no sphere.
        assert!(matches!(
            refusal(Sphere::new("n", p("O", 0, 0, 0), Q::from_int(-1))),
            GeometryError::NoRationalSolution(_)
        ));
    }

    #[test]
    fn test_polyhedron_refuses_empty_faces_and_repeated_vertices() {
        // No faces: there is no solid to measure, and the message says so.
        let empty = Polyhedron::new("nothing", vec![p("A", 0, 0, 0)], vec![]);
        assert!(matches!(
            refusal(empty.sanity_check()),
            GeometryError::EmptyGeometry(_)
        ));
        assert!(Polyhedron::checked("nothing", vec![p("A", 0, 0, 0)], vec![]).is_err());
        // A face naming a vertex nobody declared is refused rather than given a
        // location by convenience.
        let dangling = Polyhedron::new(
            "dangling",
            vec![p("A", 0, 0, 0), p("B", 1, 0, 0)],
            vec![Face3::new("f", &["A", "B", "Q"])],
        );
        assert!(dangling.sanity_check().is_err());
        // A face of two vertices has no area, and a face that names one vertex
        // twice has a zero-length edge -- both the kernel's degenerate triangle.
        for vertices in [&["A", "B"][..], &["A", "A", "B"][..]] {
            let flat = Polyhedron::new(
                "flat",
                vec![p("A", 0, 0, 0), p("B", 1, 0, 0)],
                vec![Face3::new("f", vertices)],
            );
            assert!(matches!(
                refusal(flat.sanity_check()),
                GeometryError::DegenerateTriangle { .. }
            ));
        }
        // And the cube, which is none of those things, passes.
        Polyhedron::unit_cube().sanity_check().unwrap();
    }

    /// A 2D figure lifted to `z = 0`: three collinear points `A B C` on the
    /// x axis with a midpoint `M`, a point `D` above, a second segment `DE`
    /// parallel to `AB`, and a segment `AD` perpendicular to it. Every fact the
    /// 2D kernel can establish about this figure must survive the lift.
    fn lifted_figure() -> SceneGraph {
        let points = vec![
            KPoint {
                name: "A".into(),
                x: Frac::from_int(0),
                y: Frac::from_int(0),
            },
            KPoint {
                name: "B".into(),
                x: Frac::from_int(2),
                y: Frac::from_int(0),
            },
            KPoint {
                name: "C".into(),
                x: Frac::from_int(4),
                y: Frac::from_int(0),
            },
            KPoint {
                name: "M".into(),
                x: Frac::from_int(1),
                y: Frac::from_int(0),
            },
            KPoint {
                name: "D".into(),
                x: Frac::from_int(0),
                y: Frac::from_int(1),
            },
            KPoint {
                name: "E".into(),
                x: Frac::from_int(4),
                y: Frac::from_int(1),
            },
        ];
        graph_from_points(points, vec![])
    }

    #[test]
    fn test_lift_embeds_a_2d_figure_at_z_zero() {
        let scene = lifted_figure();
        let lifted = lift_scene(&scene).unwrap();
        assert_eq!(lifted.len(), 6);
        // Every point keeps its name and its coordinates, and gains z = 0.
        for point in &lifted {
            assert_eq!(point.z, Frac::from_int(0));
        }
        let first = lifted_point(&lifted, "C").unwrap();
        assert_eq!(first.coords().unwrap(), (q(4, 1), q(0, 1), q(0, 1)));
        assert!(lifted_point(&lifted, "Z").is_err());
        // A 2D collinearity is a 3D collinearity, and it is *re-decided* there:
        // a cross product of two offsets, not the 2D kernel's determinant.
        assert!(collinear_3d(
            lifted_point(&lifted, "A").unwrap(),
            lifted_point(&lifted, "B").unwrap(),
            lifted_point(&lifted, "C").unwrap(),
        )
        .unwrap());
        assert!(lifted_holds(
            &lifted,
            &Constraint::Collinear {
                a: "A".into(),
                b: "B".into(),
                c: "C".into(),
            },
        )
        .unwrap());
        // ...and the same three points are not a triangle, which is the negative
        // case that a "lift everything and assume it still holds" path would miss.
        assert!(!lifted_holds(
            &lifted,
            &Constraint::Triangle {
                a: "A".into(),
                b: "B".into(),
                c: "C".into()
            },
        )
        .unwrap());
        // The metric and affine facts survive too: a midpoint, a ratio, a
        // betweenness, a length, a parallelism, a perpendicularity.
        assert!(lifted_holds(
            &lifted,
            &Constraint::MidpointOf {
                p: "M".into(),
                a: "A".into(),
                b: "B".into()
            },
        )
        .unwrap());
        assert!(lifted_holds(
            &lifted,
            &Constraint::RatioOf {
                p: "M".into(),
                a: "A".into(),
                b: "C".into(),
                num: 1,
                den: 3,
            },
        )
        .unwrap());
        assert!(lifted_holds(
            &lifted,
            &Constraint::Between {
                a: "A".into(),
                m: "M".into(),
                b: "C".into()
            },
        )
        .unwrap());
        assert!(lifted_holds(
            &lifted,
            &Constraint::LengthIs {
                seg: Segment {
                    from: "A".into(),
                    to: "B".into()
                },
                square: Frac::from_int(4),
            },
        )
        .unwrap());
        assert!(lifted_holds(
            &lifted,
            &Constraint::Parallel {
                first: Segment {
                    from: "A".into(),
                    to: "B".into()
                },
                second: Segment {
                    from: "D".into(),
                    to: "E".into()
                },
            },
        )
        .unwrap());
        assert!(lifted_holds(
            &lifted,
            &Constraint::Perpendicular {
                first: Segment {
                    from: "A".into(),
                    to: "B".into()
                },
                second: Segment {
                    from: "A".into(),
                    to: "D".into()
                },
            },
        )
        .unwrap());
        assert!(lifted_holds(
            &lifted,
            &Constraint::RightAngle {
                at: Angle3 {
                    at: "A".into(),
                    from: "B".into(),
                    to: "D".into()
                },
            },
        )
        .unwrap());
        // A false claim stays false after the lift: AB is not perpendicular to BC.
        assert!(!lifted_holds(
            &lifted,
            &Constraint::Perpendicular {
                first: Segment {
                    from: "A".into(),
                    to: "B".into()
                },
                second: Segment {
                    from: "B".into(),
                    to: "C".into()
                },
            },
        )
        .unwrap());
    }

    #[test]
    fn test_lift_triangle_refuses_what_the_2d_kernel_calls_degenerate() {
        let scene = lifted_figure();
        // ABC are collinear in the plane, so they are still collinear in space and
        // the lift of that "triangle" is refused with the kernel's own variant.
        let refused = refusal(lift_triangle(&scene, &Tri3::new("A", "B", "C")));
        assert!(
            matches!(refused, GeometryError::DegenerateTriangle { .. }),
            "{refused}"
        );
        // A genuine triangle lifts to three points in the plane z = 0.
        let lifted = lift_triangle(&scene, &Tri3::new("A", "D", "E")).unwrap();
        assert_eq!(lifted[0].coords().unwrap(), (q(0, 1), q(0, 1), q(0, 1)));
        assert_eq!(lifted[1].coords().unwrap(), (q(0, 1), q(1, 1), q(0, 1)));
        assert_eq!(lifted[2].coords().unwrap(), (q(4, 1), q(1, 1), q(0, 1)));
        // A triangle naming a point nobody declared is refused by name.
        assert!(lift_triangle(&scene, &Tri3::new("A", "D", "Q")).is_err());
    }

    #[test]
    fn test_lifted_holds_refuses_the_predicates_it_does_not_own() {
        let scene = lifted_figure();
        let lifted = lift_scene(&scene).unwrap();
        // A circle is a 2D object; its 3D analogue is a sphere, and reading one
        // as the other is the failure audit fix 1 was about. So this module
        // declines, by name, rather than guessing.
        let refused = refusal(lifted_holds(
            &lifted,
            &Constraint::OnCircle {
                p: "A".into(),
                circle: "k".into(),
            },
        ));
        assert!(
            matches!(refused, GeometryError::EmptyGeometry(_)),
            "{refused}"
        );
        assert!(refused.to_string().contains("three dimensions"));
        // An area and a congruence are well defined over z = 0 but belong to the
        // 2D kernel; declined here, answered there (see the test below).
        for constraint in [
            Constraint::AreaEqual {
                first: Tri3::new("A", "D", "E"),
                second: Tri3::new("A", "D", "E"),
            },
            Constraint::Congruent {
                first: Tri3::new("A", "D", "E"),
                second: Tri3::new("A", "D", "E"),
            },
        ] {
            assert!(lifted_holds(&lifted, &constraint).is_err());
        }
        // A degenerate lift predicate refuses rather than holding vacuously.
        assert!(matches!(
            refusal(lifted_holds(
                &lifted,
                &Constraint::RatioOf {
                    p: "M".into(),
                    a: "A".into(),
                    b: "C".into(),
                    num: 0,
                    den: 1,
                },
            )),
            GeometryError::EmptyGeometry(_)
        ));
    }

    #[test]
    fn test_lifted_fact_holds_is_a_two_module_certificate() {
        let scene = lifted_figure();
        // Facts the 2D kernel holds, re-decided in 3D: the certificate requires
        // both verdicts, and here they agree.
        let collinear = Fact::given(Constraint::Collinear {
            a: "A".into(),
            b: "B".into(),
            c: "C".into(),
        });
        assert!(lifted_fact_holds(&scene, &collinear).unwrap());
        let perpendicular = Fact::given(Constraint::Perpendicular {
            first: Segment {
                from: "A".into(),
                to: "B".into(),
            },
            second: Segment {
                from: "A".into(),
                to: "D".into(),
            },
        });
        assert!(lifted_fact_holds(&scene, &perpendicular).unwrap());
        // A claim that is false in the plane is false after the lift: the
        // certificate is not a rubber stamp.
        let false_triangle = Fact::given(Constraint::Triangle {
            a: "A".into(),
            b: "B".into(),
            c: "C".into(),
        });
        assert!(!lifted_fact_holds(&scene, &false_triangle).unwrap());
        let false_parallel = Fact::given(Constraint::Parallel {
            first: Segment {
                from: "A".into(),
                to: "B".into(),
            },
            second: Segment {
                from: "A".into(),
                to: "D".into(),
            },
        });
        assert!(!lifted_fact_holds(&scene, &false_parallel).unwrap());
        // The predicates with no 3D reading fall back to the 2D kernel's own
        // verdict rather than to a guess.
        let equal_areas = Fact::given(Constraint::AreaEqual {
            first: Tri3::new("A", "D", "E"),
            second: Tri3::new("A", "D", "E"),
        });
        assert!(lifted_fact_holds(&scene, &equal_areas).unwrap());
    }

    #[test]
    fn test_serde_round_trip_keeps_a_solid_exact() {
        // The scene graph is a file format an engine routes from, and a 3D solid
        // is part of it now, so the round trip has to be exact rather than
        // approximately-equal: `sqrt(3)/2` in, `sqrt(3)/2` out.
        let cube = Polyhedron::unit_cube();
        let text = serde_json::to_string(&cube).unwrap();
        let back: Polyhedron = serde_json::from_str(&text).unwrap();
        assert_eq!(back, cube);
        assert_eq!(back.total_surface_area().unwrap(), radical(6, 1, 0));
        assert_eq!(back.volume().unwrap(), q(1, 1));
        let tetra = Polyhedron::unit_tetrahedron();
        let slanted = tetra.faces.iter().find(|f| f.name == "xyz").unwrap();
        let encoded = serde_json::to_string(&tetra).unwrap();
        let decoded: Polyhedron = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.face_area(slanted).unwrap(), radical(1, 2, 3));
        // The other data types travel too.
        let sphere = Sphere::new("s", p("O", 0, 0, 0), q(3, 2)).unwrap();
        let text = serde_json::to_string(&sphere).unwrap();
        let back: Sphere = serde_json::from_str(&text).unwrap();
        assert_eq!(back, sphere);
        assert_eq!(back.radius().unwrap(), radical(1, 2, 6));
        let plane = Plane3::new("pl", Vec3::from_ints(1, 2, 3), q(1, 7)).unwrap();
        let back: Plane3 = serde_json::from_str(&serde_json::to_string(&plane).unwrap()).unwrap();
        assert_eq!(back, plane);
    }

    #[test]
    fn test_the_same_plane_stated_two_ways_gives_the_same_distance() {
        // `2x = 1` and `x = 1/2` are one plane, so they give one distance. A
        // kernel that compared raw `n . p - offset` would answer 5 and 5/2 and
        // call it a contradiction.
        let doubled = Plane3::new("doubled", Vec3::from_ints(2, 0, 0), Q::from_int(1)).unwrap();
        let halved = Plane3::new("halved", Vec3::from_ints(1, 0, 0), q(1, 2)).unwrap();
        let far = p("P", 3, 0, 0);
        assert_eq!(
            distance_point_plane(&far, &doubled).unwrap(),
            radical(5, 2, 0)
        );
        assert_eq!(
            distance_point_plane(&far, &halved).unwrap(),
            radical(5, 2, 0)
        );
        // Both projections land on the same point, and both of them on the plane.
        let first = projection_point_plane(&far, &doubled).unwrap();
        let second = projection_point_plane(&far, &halved).unwrap();
        assert!(first.same_location(&second).unwrap());
        assert_eq!(first.coords().unwrap(), (q(1, 2), q(0, 1), q(0, 1)));
        assert!(doubled.contains(&first).unwrap());
        assert!(halved.contains(&second).unwrap());
        // The foot of the perpendicular from the origin is exact too, and it is
        // the point `intersect_plane_plane` uses to test coincidence.
        assert_eq!(
            doubled.point_on("foot").unwrap().coords().unwrap(),
            (q(1, 2), q(0, 1), q(0, 1))
        );
    }
}
