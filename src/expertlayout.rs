//! Co-activation expert layout: order experts on disk so the experts that
//! fire together sit together.
//!
//! An inference engine that streams experts from storage (see
//! [`crate::peregrine`]) pays per contiguous read. If the experts a token
//! selects sit next to each other, one read covers them. If they are scattered,
//! each one is its own round trip. This module measures which experts fire
//! together and plans an order that keeps them close.
//!
//! - [`CoActivation`] accumulates an `E x E` symmetric count matrix, per-expert
//!   usage and the distinct expert sets themselves, from per-token selected
//!   sets: [`crate::routeprobe::RoutedLayer`]s from routing traces, or sets
//!   supplied directly. For MoSME, use one accumulator per box and
//!   [`CoActivation::observe_box`], which counts a box's expert set only at
//!   positions where the box itself was selected. Every box's expert router
//!   runs for every token, and an unselected box's experts never fire.
//! - [`plan_layout`] orders the experts greedily and reports the order's
//!   **locality** (the fraction of co-activation mass between experts at most
//!   `window` slots apart) for the identity order and for the plan, plus the
//!   expected number of contiguous runs per observed set.
//! - [`apply_layout`], [`read_expert`] and [`coalesced_regions`] lay blobs out
//!   in the planned order and read them back, so a plan can drive a striped
//!   file directly.
//!
//! # What this does not claim
//!
//! Locality and expected runs are properties of an order measured against the
//! observed routing. They are **not** I/O measurements: nothing here times a
//! read, and no speedup is claimed. Whether fewer runs means faster loads
//! depends on the drive, the read size and the engine. That is measured with
//! [`crate::peregrine`], not asserted here.
//!
//! # Formats are untouched
//!
//! A [`LayoutPlan`] is a separate JSON artifact. It does not change
//! [`crate::expert_index`]'s format or any checkpoint layout. A checkpoint
//! stays in its own order, and a laid-out file is a second file built from the
//! plan with [`apply_layout`].

use std::collections::BTreeMap;

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

use crate::peregrine::Region;
use crate::routeprobe::RoutedLayer;

/// Co-activation counts over `experts` experts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoActivation {
    experts: usize,
    /// Row-major `experts x experts`, symmetric, zero diagonal.
    pairs: Vec<u64>,
    usage: Vec<u64>,
    /// Every distinct observed set (sorted, deduplicated) and how often it
    /// was observed: what the expected-runs number is averaged over.
    sets: BTreeMap<Vec<usize>, u64>,
}

impl CoActivation {
    pub fn new(experts: usize) -> Result<Self> {
        anyhow::ensure!(experts > 0, "a layout needs at least one expert");
        let cells = experts
            .checked_mul(experts)
            .context("the co-activation matrix size overflows")?;
        Ok(Self {
            experts,
            pairs: vec![0; cells],
            usage: vec![0; experts],
            sets: BTreeMap::new(),
        })
    }

    pub fn experts(&self) -> usize {
        self.experts
    }

    /// How often expert `e` was selected.
    pub fn usage(&self) -> &[u64] {
        &self.usage
    }

    /// How often `a` and `b` were selected together; `0` on the diagonal and
    /// out of range.
    pub fn count(&self, a: usize, b: usize) -> u64 {
        if a == b || a >= self.experts || b >= self.experts {
            return 0;
        }
        self.pairs.get(a * self.experts + b).copied().unwrap_or(0)
    }

    /// Total co-activation mass: the sum of `count(a, b)` over `a < b`.
    pub fn total_mass(&self) -> u64 {
        let mut total = 0u64;
        for a in 0..self.experts {
            for b in a + 1..self.experts {
                total = total.saturating_add(self.count(a, b));
            }
        }
        total
    }

    /// Distinct observed sets and their multiplicities.
    pub fn sets(&self) -> impl Iterator<Item = (&[usize], u64)> {
        self.sets.iter().map(|(s, c)| (s.as_slice(), *c))
    }

    /// Record one token's selected set. Duplicates collapse; an id out of
    /// range is an error; an empty set records nothing.
    pub fn observe(&mut self, set: &[usize]) -> Result<()> {
        let mut set = set.to_vec();
        set.sort_unstable();
        set.dedup();
        if let Some(&bad) = set.iter().find(|&&e| e >= self.experts) {
            anyhow::bail!(
                "expert {bad} is out of range for a layout of {} experts",
                self.experts
            );
        }
        if set.is_empty() {
            return Ok(());
        }
        for (i, &a) in set.iter().enumerate() {
            if let Some(u) = self.usage.get_mut(a) {
                *u = u.saturating_add(1);
            }
            for &b in set.iter().skip(i + 1) {
                for cell in [a * self.experts + b, b * self.experts + a] {
                    if let Some(c) = self.pairs.get_mut(cell) {
                        *c = c.saturating_add(1);
                    }
                }
            }
        }
        let seen = self.sets.entry(set).or_insert(0);
        *seen = seen.saturating_add(1);
        Ok(())
    }

    fn check_width(&self, layer: &RoutedLayer) -> Result<()> {
        anyhow::ensure!(
            layer.experts == self.experts,
            "routed layer {} has {} experts, this layout {}",
            layer.name,
            layer.experts,
            self.experts
        );
        Ok(())
    }

    /// Record every position's selected set from a traced routed layer.
    pub fn observe_layer(&mut self, layer: &RoutedLayer) -> Result<()> {
        self.observe_window(layer, 1)
    }

    /// Record the union of the selected sets of every `window` consecutive
    /// positions: the experts a decode window of that many tokens touches.
    /// `window = 1` is [`Self::observe_layer`]. A sequence shorter than the
    /// window contributes its whole union once.
    ///
    /// Useful for trunk layers, which expose only the top-1 per token. A
    /// singleton set has no pairs, so per-token observation of a top-1 layer
    /// records usage and nothing else.
    pub fn observe_window(&mut self, layer: &RoutedLayer, window: usize) -> Result<()> {
        anyhow::ensure!(window > 0, "a window must cover at least one position");
        self.check_width(layer)?;
        let n = layer.selected.len();
        if n == 0 {
            return Ok(());
        }
        let starts = n.saturating_sub(window) + 1;
        for start in 0..starts {
            let union: Vec<usize> = layer
                .selected
                .iter()
                .skip(start)
                .take(window)
                .flatten()
                .copied()
                .collect();
            self.observe(&union)?;
        }
        Ok(())
    }

    /// Record one box's expert sets, only at positions where the box was
    /// selected. `boxes` is the box-level routed layer, `experts` that box's
    /// expert-level layer (this accumulator's width), `box_idx` its index.
    pub fn observe_box(
        &mut self,
        boxes: &RoutedLayer,
        box_idx: usize,
        experts: &RoutedLayer,
    ) -> Result<()> {
        self.check_width(experts)?;
        anyhow::ensure!(
            box_idx < boxes.experts,
            "box {box_idx} is out of range for a box router of {} boxes",
            boxes.experts
        );
        anyhow::ensure!(
            boxes.selected.len() == experts.selected.len(),
            "the box layer covers {} positions, the expert layer {}",
            boxes.selected.len(),
            experts.selected.len()
        );
        for (chosen, set) in boxes.selected.iter().zip(&experts.selected) {
            if chosen.contains(&box_idx) {
                self.observe(set)?;
            }
        }
        Ok(())
    }
}

/// An expert order on disk, with what it achieves on the routing it was
/// planned from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayoutPlan {
    pub experts: usize,
    /// Distance (in slots) within which co-activated experts count as local.
    pub window: usize,
    /// `order[slot]` is the expert stored in that slot.
    pub order: Vec<usize>,
    /// `inverse[expert]` is the slot that expert is stored in.
    pub inverse: Vec<usize>,
    /// Locality of the identity order (expert `e` in slot `e`).
    pub locality_before: f64,
    /// Locality of `order`. Never below `locality_before` (see
    /// [`plan_layout`]).
    pub locality_after: f64,
    /// Mean number of contiguous slot runs an observed set occupies, under
    /// the identity order and under `order`. One run is one contiguous read.
    pub runs_before: f64,
    pub runs_after: f64,
    /// Whether the greedy order scored below the identity and the identity
    /// was kept instead.
    pub kept_identity: bool,
}

impl LayoutPlan {
    /// Reject a plan that is not a bijection over `0..experts`, whose
    /// `inverse` disagrees with its `order`, or whose numbers are not
    /// numbers. Run on every plan read from disk.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.experts > 0, "a plan over zero experts");
        anyhow::ensure!(self.window > 0, "a plan with a zero window");
        anyhow::ensure!(
            self.order.len() == self.experts && self.inverse.len() == self.experts,
            "a plan over {} experts has an order of {} and an inverse of {}",
            self.experts,
            self.order.len(),
            self.inverse.len()
        );
        let mut seen = vec![false; self.experts];
        for (slot, &e) in self.order.iter().enumerate() {
            let mark = seen
                .get_mut(e)
                .with_context(|| format!("slot {slot} holds expert {e}, out of range"))?;
            anyhow::ensure!(!*mark, "expert {e} occupies more than one slot");
            *mark = true;
            let back = self
                .inverse
                .get(e)
                .with_context(|| format!("no inverse entry for expert {e}"))?;
            anyhow::ensure!(
                *back == slot,
                "inverse[{e}] = {back}, but expert {e} is stored in slot {slot}"
            );
        }
        for (name, v, hi) in [
            ("locality_before", self.locality_before, 1.0),
            ("locality_after", self.locality_after, 1.0),
            ("runs_before", self.runs_before, f64::INFINITY),
            ("runs_after", self.runs_after, f64::INFINITY),
        ] {
            anyhow::ensure!(
                v.is_finite() && (0.0..=hi).contains(&v),
                "{name} = {v} is not a valid measurement"
            );
        }
        Ok(())
    }

    /// The slot expert `expert` is stored in.
    pub fn slot_of(&self, expert: usize) -> Result<usize> {
        self.inverse.get(expert).copied().with_context(|| {
            format!(
                "expert {expert} is out of range for a plan of {}",
                self.experts
            )
        })
    }

    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Parse and [`Self::validate`]: a plan that does not validate is never
    /// returned.
    pub fn from_json(text: &str) -> Result<Self> {
        let plan: Self = serde_json::from_str(text).context("parse a layout plan")?;
        plan.validate()?;
        Ok(plan)
    }
}

/// Fraction of co-activation mass between experts at most `window` slots
/// apart under `inverse`. `0.0` when there is no mass to localize.
pub fn locality(co: &CoActivation, inverse: &[usize], window: usize) -> f64 {
    let total = co.total_mass();
    if total == 0 {
        return 0.0;
    }
    let mut near = 0u64;
    for a in 0..co.experts() {
        for b in a + 1..co.experts() {
            let (Some(&sa), Some(&sb)) = (inverse.get(a), inverse.get(b)) else {
                continue;
            };
            if sa.abs_diff(sb) <= window {
                near = near.saturating_add(co.count(a, b));
            }
        }
    }
    near as f64 / total as f64
}

/// Number of maximal runs of consecutive slots the experts of `set` occupy.
pub fn contiguous_runs(set: &[usize], inverse: &[usize]) -> Result<usize> {
    let mut slots = set
        .iter()
        .map(|&e| {
            inverse
                .get(e)
                .copied()
                .with_context(|| format!("expert {e} has no slot"))
        })
        .collect::<Result<Vec<_>>>()?;
    slots.sort_unstable();
    slots.dedup();
    Ok(match slots.len() {
        0 => 0,
        _ => 1 + slots.windows(2).filter(|w| w[1] != w[0] + 1).count(),
    })
}

/// Mean [`contiguous_runs`] over every observed set, weighted by how often it
/// was observed; `0.0` with nothing observed.
pub fn expected_runs(co: &CoActivation, inverse: &[usize]) -> Result<f64> {
    let mut weighted = 0.0f64;
    let mut total = 0.0f64;
    for (set, count) in co.sets() {
        let c = count as f64;
        weighted += c * contiguous_runs(set, inverse)? as f64;
        total += c;
    }
    Ok(if total > 0.0 { weighted / total } else { 0.0 })
}

/// Plan an expert order from `co`.
///
/// Greedy and deterministic: start at the most-used expert (ties: lowest
/// id), then repeatedly append the unplaced expert with the highest total
/// co-activation with the last `window` placed experts, breaking ties by
/// higher usage, then lower id. If that order's locality is below the
/// identity order's, the identity is kept (`kept_identity`), so a plan
/// **never lowers locality**. A greedy pass is not optimal and can do worse
/// than an order that is already good.
pub fn plan_layout(co: &CoActivation, window: usize) -> Result<LayoutPlan> {
    anyhow::ensure!(window > 0, "a layout window must be at least one slot");
    let e = co.experts();
    let usage = co.usage();
    let use_of = |x: usize| usage.get(x).copied().unwrap_or(0);

    let mut placed = vec![false; e];
    let mut order = Vec::with_capacity(e);
    while order.len() < e {
        let recent: &[usize] = order
            .get(order.len().saturating_sub(window)..)
            .unwrap_or(&[]);
        let mut best: Option<(u64, u64, usize)> = None;
        for (x, &done) in placed.iter().enumerate() {
            if done {
                continue;
            }
            let score = recent
                .iter()
                .fold(0u64, |acc, &q| acc.saturating_add(co.count(x, q)));
            let better = match best {
                None => true,
                // Higher score, then higher usage; `x` ascends, so a full tie
                // keeps the earlier (lower) id.
                Some((s, u, _)) => (score, use_of(x)) > (s, u),
            };
            if better {
                best = Some((score, use_of(x), x));
            }
        }
        let (_, _, next) = best.context("no unplaced expert left while the order is short")?;
        if let Some(mark) = placed.get_mut(next) {
            *mark = true;
        }
        order.push(next);
    }

    let identity: Vec<usize> = (0..e).collect();
    let mut inverse = vec![0usize; e];
    for (slot, &x) in order.iter().enumerate() {
        if let Some(s) = inverse.get_mut(x) {
            *s = slot;
        }
    }
    let locality_before = locality(co, &identity, window);
    let greedy_locality = locality(co, &inverse, window);
    let kept_identity = greedy_locality < locality_before;
    if kept_identity {
        order.clone_from(&identity);
        inverse.clone_from(&identity);
    }
    let plan = LayoutPlan {
        experts: e,
        window,
        locality_before,
        locality_after: if kept_identity {
            locality_before
        } else {
            greedy_locality
        },
        runs_before: expected_runs(co, &identity)?,
        runs_after: expected_runs(co, &inverse)?,
        order,
        inverse,
        kept_identity,
    };
    plan.validate()?;
    Ok(plan)
}

/// One slot's `(offset, len)` in a laid-out buffer or file.
pub type Extent = (u64, usize);

/// Concatenate `blobs` (indexed by expert id) in the plan's slot order.
/// Returns the laid-out bytes and the slot table: `extents[slot]` is the
/// `(offset, len)` of expert `plan.order[slot]`. Zero-length blobs are legal
/// and occupy an empty extent.
pub fn apply_layout(blobs: &[Vec<u8>], plan: &LayoutPlan) -> Result<(Vec<u8>, Vec<Extent>)> {
    plan.validate()?;
    anyhow::ensure!(
        blobs.len() == plan.experts,
        "{} blobs for a plan over {} experts",
        blobs.len(),
        plan.experts
    );
    let total = blobs
        .iter()
        .try_fold(0usize, |acc, b| acc.checked_add(b.len()))
        .context("the laid-out size overflows")?;
    let mut bytes = Vec::with_capacity(total);
    let mut extents = Vec::with_capacity(plan.experts);
    for &e in &plan.order {
        let blob = blobs
            .get(e)
            .with_context(|| format!("no blob for expert {e}"))?;
        let offset = u64::try_from(bytes.len()).context("offset does not fit in u64")?;
        extents.push((offset, blob.len()));
        bytes.extend_from_slice(blob);
    }
    Ok((bytes, extents))
}

/// Expert `expert`'s bytes in a buffer laid out by [`apply_layout`].
pub fn read_expert<'a>(
    bytes: &'a [u8],
    extents: &[Extent],
    plan: &LayoutPlan,
    expert: usize,
) -> Result<&'a [u8]> {
    anyhow::ensure!(
        extents.len() == plan.experts,
        "a slot table of {} entries for a plan over {} experts",
        extents.len(),
        plan.experts
    );
    let slot = plan.slot_of(expert)?;
    let &(offset, len) = extents
        .get(slot)
        .with_context(|| format!("no extent for slot {slot}"))?;
    let start = usize::try_from(offset).context("offset does not fit in usize")?;
    let end = start.checked_add(len).context("extent end overflows")?;
    bytes.get(start..end).with_context(|| {
        format!(
            "expert {expert} (slot {slot}) spans {start}..{end}, past a buffer of {}",
            bytes.len()
        )
    })
}

/// The reads that fetch every expert in `set` from a laid-out file whose
/// first byte sits at `base`: one [`Region`] per maximal byte-contiguous run,
/// in file order, zero-length experts omitted. The region count is the number
/// of contiguous reads the set costs. It is a count, not a time.
pub fn coalesced_regions(
    extents: &[Extent],
    plan: &LayoutPlan,
    set: &[usize],
    base: u64,
) -> Result<Vec<Region>> {
    anyhow::ensure!(
        extents.len() == plan.experts,
        "a slot table of {} entries for a plan over {} experts",
        extents.len(),
        plan.experts
    );
    let mut spans = Vec::with_capacity(set.len());
    for &e in set {
        let slot = plan.slot_of(e)?;
        let &(offset, len) = extents
            .get(slot)
            .with_context(|| format!("no extent for slot {slot}"))?;
        if len > 0 {
            spans.push((offset, len));
        }
    }
    spans.sort_unstable();
    spans.dedup();
    let mut regions: Vec<Region> = Vec::with_capacity(spans.len());
    for (offset, len) in spans {
        let start = base
            .checked_add(offset)
            .context("region offset overflows")?;
        match regions.last_mut() {
            Some(last) if last.end() == start => {
                last.len = last
                    .len
                    .checked_add(len)
                    .context("region length overflows")?;
            }
            _ => regions.push(Region::new(start, len)),
        }
    }
    Ok(regions)
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
    use crate::routing::RouterKind;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng};

    fn layer(name: &str, experts: usize, selected: Vec<Vec<usize>>) -> RoutedLayer {
        RoutedLayer {
            name: name.into(),
            kind: RouterKind::Ffn,
            experts,
            selected,
            probs: None,
        }
    }

    /// `blocks` groups of `size` experts under a seeded shuffle of ids; every
    /// pair inside a block co-fires, nothing crosses blocks.
    fn planted(blocks: usize, size: usize, seed: u64) -> (CoActivation, Vec<Vec<usize>>) {
        let e = blocks * size;
        let mut ids: Vec<usize> = (0..e).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        ids.shuffle(&mut rng);
        let groups: Vec<Vec<usize>> = ids.chunks(size).map(<[usize]>::to_vec).collect();
        let mut co = CoActivation::new(e).unwrap();
        for g in &groups {
            co.observe(g).unwrap();
            for _ in 0..20 {
                let a = g[rng.random_range(0..size)];
                let b = g[rng.random_range(0..size)];
                co.observe(&[a, b]).unwrap();
            }
        }
        (co, groups)
    }

    #[test]
    fn co_activation_counts_symmetrically_and_rejects_bad_ids() {
        let mut co = CoActivation::new(4).unwrap();
        co.observe(&[2, 0, 2]).unwrap();
        co.observe(&[0, 2, 3]).unwrap();
        co.observe(&[]).unwrap();
        assert_eq!(co.count(0, 2), 2);
        assert_eq!(co.count(2, 0), 2);
        assert_eq!(co.count(2, 3), 1);
        assert_eq!(co.count(1, 1), 0);
        assert_eq!(co.count(9, 0), 0);
        assert_eq!(co.usage(), &[2, 0, 2, 1]);
        assert_eq!(co.total_mass(), 4);
        assert_eq!(co.sets().count(), 2);
        assert!(co.observe(&[4]).is_err());
        assert!(CoActivation::new(0).is_err());

        let mut windowed = CoActivation::new(3).unwrap();
        let l = layer("trunk.sparse[0]", 3, vec![vec![0], vec![1], vec![2]]);
        windowed.observe_window(&l, 2).unwrap();
        assert_eq!(windowed.count(0, 1), 1);
        assert_eq!(windowed.count(1, 2), 1);
        assert_eq!(windowed.count(0, 2), 0);
        let mut single = CoActivation::new(3).unwrap();
        single.observe_layer(&l).unwrap();
        assert_eq!(single.total_mass(), 0, "top-1 sets carry no pairs");
        assert!(single.observe_layer(&layer("x", 4, vec![])).is_err());
    }

    #[test]
    fn box_observation_counts_only_where_the_box_fired() {
        let boxes = layer("probe.box", 2, vec![vec![0], vec![1], vec![0]]);
        let experts = layer(
            "probe.box[0].expert",
            3,
            vec![vec![0, 1], vec![1, 2], vec![0, 2]],
        );
        let mut co = CoActivation::new(3).unwrap();
        co.observe_box(&boxes, 0, &experts).unwrap();
        assert_eq!(co.count(0, 1), 1);
        assert_eq!(co.count(0, 2), 1);
        assert_eq!(co.count(1, 2), 0, "box 0 was not selected at position 1");
        assert!(co.observe_box(&boxes, 2, &experts).is_err());
    }

    #[test]
    fn a_planted_structure_is_recovered_contiguously() {
        let (co, groups) = planted(4, 3, 11);
        let plan = plan_layout(&co, 2).unwrap();
        println!(
            "planted 4x3: locality {:.3} -> {:.3}, expected runs {:.3} -> {:.3}, order {:?}",
            plan.locality_before,
            plan.locality_after,
            plan.runs_before,
            plan.runs_after,
            plan.order
        );
        assert!(!plan.kept_identity);
        assert!(plan.locality_after >= plan.locality_before);
        assert!(
            plan.locality_before < 1.0,
            "the shuffle must scatter the blocks"
        );
        assert_eq!(plan.locality_after, 1.0);
        for g in &groups {
            let slots: Vec<usize> = g.iter().map(|&e| plan.inverse[e]).collect();
            let (lo, hi) = (slots.iter().min().unwrap(), slots.iter().max().unwrap());
            assert_eq!(hi - lo + 1, g.len(), "block {g:?} sits in slots {slots:?}");
            assert_eq!(contiguous_runs(g, &plan.inverse).unwrap(), 1);
        }
        // Not 1.0: a block of three in slots s..s+2 is one run, but an
        // observed pair of its two ends is still two.
        assert!(plan.runs_before > plan.runs_after);
        // Deterministic: the same counts plan the same order.
        assert_eq!(plan_layout(&co, 2).unwrap(), plan);
    }

    #[test]
    fn an_already_good_order_is_kept_rather_than_lowered() {
        // A chain 0-1-2-3 in identity order is perfectly local at window 1.
        // Greedy starts at the most-used expert (1), goes to 2 (more used than
        // 0), then 3, then 0: two of three links local. The plan keeps the
        // identity instead.
        let mut co = CoActivation::new(4).unwrap();
        for _ in 0..10 {
            co.observe(&[0, 1]).unwrap();
            co.observe(&[1, 2]).unwrap();
            co.observe(&[2, 3]).unwrap();
        }
        for _ in 0..5 {
            co.observe(&[1]).unwrap();
            co.observe(&[2]).unwrap();
        }
        co.observe(&[1]).unwrap();
        let plan = plan_layout(&co, 1).unwrap();
        assert!(plan.kept_identity, "{plan:?}");
        assert_eq!(plan.order, vec![0, 1, 2, 3]);
        assert_eq!(plan.locality_before, 1.0);
        assert_eq!(plan.locality_after, 1.0);
        assert!(plan_layout(&co, 0).is_err());
    }

    #[test]
    fn laid_out_reads_are_byte_identical_and_coalesce() {
        let (co, _) = planted(3, 2, 5);
        let plan = plan_layout(&co, 1).unwrap();
        let mut rng = StdRng::seed_from_u64(9);
        let blobs: Vec<Vec<u8>> = (0..plan.experts)
            .map(|i| {
                let len = if i % 3 == 0 {
                    0
                } else {
                    rng.random_range(1..40)
                };
                (0..len).map(|_| rng.random()).collect()
            })
            .collect();
        let (bytes, extents) = apply_layout(&blobs, &plan).unwrap();
        assert_eq!(bytes.len(), blobs.iter().map(Vec::len).sum::<usize>());
        for (e, blob) in blobs.iter().enumerate() {
            assert_eq!(
                read_expert(&bytes, &extents, &plan, e).unwrap(),
                blob.as_slice()
            );
        }
        assert!(read_expert(&bytes, &extents, &plan, plan.experts).is_err());
        assert!(read_expert(&bytes[..bytes.len() / 2], &extents, &plan, plan.order[5]).is_err());
        assert!(apply_layout(&blobs[1..], &plan).is_err());

        // A whole slot range reads as one region; a gap splits it.
        let all: Vec<usize> = plan.order.clone();
        let one = coalesced_regions(&extents, &plan, &all, 100).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0], Region::new(100, bytes.len()));
        let gapped = [plan.order[1], plan.order[4]];
        let regions = coalesced_regions(&extents, &plan, &gapped, 0).unwrap();
        let nonempty = gapped.iter().filter(|&&e| !blobs[e].is_empty()).count();
        assert!(regions.len() <= nonempty);
        assert_eq!(
            regions.iter().map(|r| r.len).sum::<usize>(),
            gapped.iter().map(|&e| blobs[e].len()).sum::<usize>(),
            "the regions cover exactly the requested experts' bytes"
        );
        for r in &regions {
            let start = usize::try_from(r.offset).unwrap();
            assert!(start + r.len <= bytes.len());
        }
    }

    #[test]
    fn validate_rejects_every_malformed_plan() {
        let (co, _) = planted(2, 2, 3);
        let good = plan_layout(&co, 1).unwrap();
        good.validate().unwrap();
        let json = good.to_json().unwrap();
        assert_eq!(LayoutPlan::from_json(&json).unwrap(), good);

        let mut dup = good.clone();
        dup.order[1] = dup.order[0];
        assert!(dup.validate().is_err(), "not a bijection");
        let mut out_of_range = good.clone();
        out_of_range.order[0] = 99;
        assert!(out_of_range.validate().is_err());
        let mut short = good.clone();
        short.order.pop();
        assert!(short.validate().is_err());
        let mut wrong_inverse = good.clone();
        wrong_inverse.inverse.swap(0, 1);
        assert!(wrong_inverse.validate().is_err());
        let mut nan = good.clone();
        nan.locality_after = f64::NAN;
        assert!(nan.validate().is_err());
        let mut zero_window = good.clone();
        zero_window.window = 0;
        assert!(zero_window.validate().is_err());

        let tampered = json.replacen("\"window\": 1", "\"window\": 0", 1);
        assert_ne!(tampered, json);
        assert!(LayoutPlan::from_json(&tampered).is_err());
    }

    #[test]
    fn a_traced_routing_plans_a_valid_layout() {
        // End to end through the probe's own control router: trace, observe
        // with a decode window, plan, lay out, read back.
        use crate::routeprobe::{marker_tokens, PooledControlRouter, Pooling, RoutingTracer};
        let router =
            PooledControlRouter::seeded(crate::tokenizer::VOCAB_SIZE, 8, 8, 2, Pooling::Prefix, 4)
                .unwrap();
        let trace = router
            .routing_trace(&marker_tokens("fn route(x: u8) -> u8 { x }"))
            .unwrap();
        let mut co = CoActivation::new(8).unwrap();
        for l in &trace.layers {
            co.observe_window(l, 2).unwrap();
        }
        let plan = plan_layout(&co, 2).unwrap();
        println!(
            "control router: locality {:.3} -> {:.3}, runs {:.3} -> {:.3}",
            plan.locality_before, plan.locality_after, plan.runs_before, plan.runs_after
        );
        assert!(plan.locality_after >= plan.locality_before);
        let blobs: Vec<Vec<u8>> = (0..8u8).map(|i| vec![i; usize::from(i)]).collect();
        let (bytes, extents) = apply_layout(&blobs, &plan).unwrap();
        for (e, blob) in blobs.iter().enumerate() {
            assert_eq!(
                read_expert(&bytes, &extents, &plan, e).unwrap(),
                blob.as_slice()
            );
        }
    }
}
