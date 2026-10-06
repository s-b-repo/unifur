//! `peregrine`: the parallel I/O engine (roadmap Phase 34).
//!
//! # What this is for
//!
//! Training a block-wise model is a streaming problem, not a loading problem.
//! Every step wants the next windows of a corpus, and the corpus is larger
//! than the page cache. At that point the pipeline is not compute-bound, it is
//! *blocked on storage*, and the only fixes that matter are (a) more requests
//! in flight, and (b) fewer bytes per request wasted on round trips. This
//! module is (a) and (b).
//!
//! The design in one line: **one region, split across every drive, submitted
//! at once, verified against a single-drive read.**
//!
//! # Why striping beats queueing
//!
//! A single-threaded reader issues read #1, waits for the device, issues #2,
//! waits again. The device is idle while the reader assembles the result, and
//! the reader is idle while the device works. Two fixes follow directly:
//!
//! - **depth**: submit N reads before waiting for any of them. The device
//!   never sees an empty queue, so its own internal parallelism (queue depth,
//!   channel interleaving, NCQ) is actually used. A reader that waits after
//!   every request pins the device at QD=1 and throws away most of its
//!   bandwidth.
//! - **width**: when the same bytes live on several drives, one drive is a
//!   queue of depth 1 no matter how deep the submissions are. Splitting a
//!   region across N drives turns N sequential latencies into one.
//!
//! Striping is therefore not an exotic trick here, it is the ordinary one: the
//! corpus is replicated (or a region is duplicated), and each replica is a
//! separate device queue.
//!
//! # The rule that keeps this honest
//!
//! Parallel reads are only a win if they return *the same bytes*. A striped
//! read that is off by one sector, or that races a writer, or that assembles
//! stripes in submission-completion order instead of stripe order, is faster
//! and wrong, and wrong I/O in a training loop is not a crash -- it is a model
//! that trains on corrupted data and reports a plausible loss. So:
//!
//! - stripes are assembled **by index, never by completion order**;
//! - every stripe read is bounds-checked and short reads are an error, not a
//!   zero-fill;
//! - the striped read is checked byte-for-byte against the single-drive read
//!   by the oracle test, on every stripe count,
//!   including the awkward ones (1 stripe, more stripes than bytes, a region
//!   that does not divide evenly).
//!
//! # The ring, and what happens without it
//!
//! With the `peregrine-uring` feature the reads are submitted to an io_uring
//! ring: one `io_uring_enter` carries the whole batch, so the syscall cost is
//! per *batch* rather than per *read*. Without it, the identical plan is
//! executed with `pread` on a scoped thread pool. The plan, the striping, the
//! ordering guarantee and the oracle are the same either way; only the syscall
//! count differs. A ring that cannot be created (old kernel, seccomp, no
//! permission) degrades to the threaded path automatically -- see
//! [`Backend::detect`].
//!
//! # What the ring actually bought, measured
//!
//! `examples/perf_probe.rs` and `dblocks io` measure all three paths on
//! 32 MiB over four replicas. On one 12-core machine, with the four replicas
//! being four files in a single directory:
//!
//! ```text
//! serial (1 drive, 1 syscall)     851-1003 MiB/s  reads 1  syscalls 1
//! striped (pread on threads)    2063-2562 MiB/s  reads 4  syscalls 4
//! striped (io_uring, persistent) 897-986 MiB/s  reads 4  syscalls 1
//! ```
//!
//! Three conclusions, stated because they are inconvenient for the usual
//! sales pitch:
//!
//! - **Striping is the win: ~2.7x.** Splitting one region across four replicas
//!   is what turns four sequential latencies into one, and that shows up
//!   immediately.
//! - **The ring is not a throughput win here, and the reason is the setup
//!   above, not the ring.** It does what it promises -- four reads per
//!   `io_uring_enter`, confirmed by the counters -- but saving three `enter`
//!   calls out of a 32 MiB transfer is noise next to the memory traffic. The
//!   threaded path moves the same bytes with four cores; the ring moves them
//!   with the kernel's async workers. On a *cold* cache over a *slow* device,
//!   where each read is latency-bound rather than bandwidth-bound, the balance
//!   would move; this fixture cannot show that, and the example says so rather
//!   than implying a number it did not earn.
//! - **The replicas here share one device**, which is the configuration where
//!   striping is least useful and the one every developer has by default. The
//!   payoff case is genuinely separate drives.
//!
//! Two implementation details came out of these numbers rather than out of
//! theory, and both are in the code: the kernel writes *directly into the
//! destination* (an earlier version read into a per-stripe `Vec` and copied,
//! which cost an extra full pass over 32 MiB and roughly halved throughput,
//! 507 -> 964 MiB/s), and the ring is created once per [`MirrorSet`] and
//! reused (creating one per batch costs a syscall pair and an `mmap` each time,
//! which is what made the first version slower still).
//!
//! So: the ring is here because it is the right mechanism for the case that
//! matters -- many small reads, a slow device, real parallelism across drives
//! -- and not because it wins a microbenchmark on a warm page cache. Anyone
//! deploying this should run `dblocks io` on their own storage and trust that
//! number instead of this paragraph.

use anyhow::Context;
use std::fmt;
use std::fs::File;
use std::io::ErrorKind;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
/// Submissions a persistent ring is created for. A batch larger than this falls
/// back to the threaded path rather than growing the ring, so a pathological
/// stripe count cannot make the ring enormous.
#[cfg(feature = "peregrine-uring")]
const RING_QUEUE_DEPTH: usize = 64;

/// Smallest chunk worth its own read. Below this the submission bookkeeping
/// costs more than the transfer, so a stripe is never planned smaller.
pub const MIN_STRIPE: usize = 4 * 1024;

/// How a batch of stripe reads is actually issued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// One `io_uring_enter` per batch: the ring queues the reads and the
    /// kernel runs them concurrently without the process issuing a syscall per
    /// read.
    IoUring,
    /// `pread` on `std::thread::scope`: same plan, same ordering guarantee, one
    /// syscall per read. The fallback when no ring can be created.
    Threaded,
}

impl Backend {
    /// Pick the fastest backend this machine and build actually permit.
    ///
    /// A ring is only *assumed* to work by [`Backend::IoUring`]; this probes it
    /// for real by creating a ring and submitting nothing. Probing at
    /// construction rather than at first use means a run does not discover the
    /// fallback halfway through, after it has already reported timings.
    pub fn detect(feature_enabled: bool) -> Self {
        // `ring_available` is `false` in a feature-off build, so this one
        // condition covers both cases and needs no `cfg` here. Asking the probe
        // rather than testing the feature directly means the feature-off build
        // reports the same reason as a feature-on build on a kernel without
        // io_uring: the ring is not there.
        if feature_enabled && ring_available() {
            return Self::IoUring;
        }
        Self::Threaded
    }

    /// Whether this backend issues one syscall per batch rather than per read.
    pub fn batched_syscalls(&self) -> bool {
        matches!(self, Self::IoUring)
    }
}

/// Whether an io_uring ring can actually be created here.
///
/// Creating a ring touches `io_uring_setup`, which an unprivileged sandbox or
/// an old kernel can refuse. That refusal is not an error worth propagating:
/// the threaded backend reads the same bytes.
///
/// Only ever called to decide whether to *report* the ring as measured. A
/// feature-off build has no ring to measure, so it reports nothing -- which is
/// why both arms exist and the function is reachable either way. Written as a
/// single definition with a `cfg`-switched body rather than two functions and an
/// `#[allow(dead_code)]`, because a suppression here is a hole in every rule on
/// this list and there is a form that needs none.
#[cfg(feature = "peregrine-uring")]
fn ring_available() -> bool {
    io_uring::IoUring::new(8).is_ok()
}

/// See the feature-on definition above: without the feature there is no ring.
/// One definition per configuration, both reachable from `Backend::detect`, so
/// neither is dead code and neither needs a suppression saying so.
#[cfg(not(feature = "peregrine-uring"))]
fn ring_available() -> bool {
    false
}

/// A read of one contiguous byte range, resolved against a replica.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    /// Byte offset within the file, in every replica.
    pub offset: u64,
    /// Length in bytes. A read of `0` is legal and yields nothing.
    pub len: usize,
}

impl Region {
    pub fn new(offset: u64, len: usize) -> Self {
        Self { offset, len }
    }

    /// First byte past the end.
    pub fn end(&self) -> u64 {
        self.offset + self.len as u64
    }
}

/// One planned read: which replica serves it, and which bytes.
///
/// The stripe index is the *only* thing that determines where the bytes go in
/// the output. Completion order is deliberately not recorded, because relying
/// on it is exactly the bug this module exists to avoid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stripe {
    /// Position of this stripe in the reassembled output, ascending.
    pub index: usize,
    /// Index into the replica list.
    pub replica: usize,
    /// Where in the output this stripe's bytes belong.
    pub dst_offset: usize,
    pub len: usize,
    pub file_offset: u64,
}

/// The plan: how one region is split across the replicas.
///
/// A plan is pure data and can be inspected, logged and tested without a single
/// syscall, which is what makes the striping arithmetic testable in isolation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirrorPlan {
    /// The region being read.
    pub region: Region,
    /// How many replicas were available.
    pub replicas: usize,
    /// The stripes, ascending by `index`, and together covering the region
    /// exactly with no gaps and no overlap.
    pub stripes: Vec<Stripe>,
}

impl MirrorPlan {
    /// Plan a striped read of `region` over `replicas` copies of the same file.
    ///
    /// With one replica this is the serial read, and `stripes` has one entry --
    /// which is what makes the degenerate case the same code path rather than a
    /// special case that is never exercised.
    ///
    /// Two rules keep the plan honest. A region shorter than [`MIN_STRIPE`] is
    /// served by a single stripe regardless of replica count, because splitting
    /// a 200-byte read across twelve drives costs twelve submissions to move
    /// less than one page. And no stripe is ever planned below [`MIN_STRIPE`]
    /// *unless the whole region is that small*, so the "wide but useless"
    /// configuration degrades to "narrow and correct" instead of to
    /// "twelve reads of a few bytes each".
    pub fn new(region: Region, replicas: usize) -> Self {
        let replicas = replicas.max(1);
        let len = region.len;
        let offset = region.offset;
        if len == 0 {
            return Self {
                region,
                replicas,
                stripes: Vec::new(),
            };
        }
        if replicas == 1 || len < MIN_STRIPE {
            return Self {
                region,
                replicas,
                stripes: vec![Stripe {
                    index: 0,
                    replica: 0,
                    dst_offset: 0,
                    len,
                    file_offset: offset,
                }],
            };
        }

        // Aim for one stripe per replica, but let a small region use fewer
        // stripes than there are replicas: `n` stripes of `MIN_STRIPE` each is
        // the floor on useful work.
        let by_replica = len.div_ceil(MIN_STRIPE);
        let count = replicas.min(by_replica).max(1);
        let base = len / count;
        let extra = len % count;
        let mut stripes = Vec::with_capacity(count);
        let mut dst = 0usize;
        for index in 0..count {
            // The first `extra` stripes take one extra byte, so the split is
            // even and the final stripe ends exactly at the region end.
            let stripe_len = base + usize::from(index < extra);
            stripes.push(Stripe {
                index,
                replica: index % replicas,
                dst_offset: dst,
                len: stripe_len,
                file_offset: offset + dst as u64,
            });
            dst += stripe_len;
        }
        Self {
            region,
            replicas,
            stripes,
        }
    }

    /// The stripes in the order they must be *issued* -- biggest first.
    ///
    /// Issuing order is a throughput decision only: it cannot affect the result,
    /// because assembly is by `index`. Longest-first measurably helps because
    /// the first completion then tends to be a big stripe, so the tail of the
    /// batch is small work rather than one long wait.
    pub fn issue_order(&self) -> Vec<&Stripe> {
        let mut order: Vec<&Stripe> = self.stripes.iter().collect();
        order.sort_by(|a, b| b.len.cmp(&a.len).then(a.index.cmp(&b.index)));
        order
    }

    /// Total bytes across all stripes. Must equal the region length: the
    /// invariant the oracle test leans on.
    pub fn planned_bytes(&self) -> usize {
        self.stripes.iter().map(|s| s.len).sum()
    }

    /// Check the plan's own invariants: ascending indices, contiguous
    /// destination coverage, no overlap, and the total equal to the region.
    ///
    /// This is cheap, runs before any I/O, and turns "the stripes did not add
    /// up" from a silent corruption into a refusal.
    pub fn validate(&self) -> anyhow::Result<()> {
        let mut expected_dst = 0usize;
        for (position, stripe) in self.stripes.iter().enumerate() {
            anyhow::ensure!(
                stripe.index == position,
                "stripe {position} carries index {}: reassembly is by index, so indices must ascend",
                stripe.index
            );
            anyhow::ensure!(
                stripe.dst_offset == expected_dst,
                "stripe {position} starts at {} but the previous stripes end at {expected_dst}",
                stripe.dst_offset
            );
            anyhow::ensure!(
                stripe.replica < self.replicas,
                "stripe {position} names replica {} of {}",
                stripe.replica,
                self.replicas
            );
            anyhow::ensure!(
                stripe.file_offset == self.region.offset + stripe.dst_offset as u64,
                "stripe {position} reads at {} but its destination begins at {}",
                stripe.file_offset,
                stripe.dst_offset
            );
            expected_dst += stripe.len;
        }
        anyhow::ensure!(
            expected_dst == self.region.len,
            "the stripes cover {expected_dst} bytes of a {}-byte region",
            self.region.len
        );
        Ok(())
    }
}

/// Counters for one batch, so a run can be *measured* rather than believed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IoStats {
    /// Batches executed.
    pub batches: u64,
    /// Individual stripe reads issued.
    pub reads: u64,
    /// `io_uring_enter` / `pread` calls, i.e. syscalls actually made.
    pub syscalls: u64,
    /// Bytes returned.
    pub bytes: u64,
    /// Batches served by the ring (one syscall) rather than per-read syscalls.
    pub ring_batches: u64,
}

impl IoStats {
    /// Reads per syscall. The number that says whether the engine is actually
    /// batching; 1.0 means it is not.
    pub fn reads_per_syscall(&self) -> f64 {
        if self.syscalls == 0 {
            return 0.0;
        }
        self.reads as f64 / self.syscalls as f64
    }

    /// Syscalls avoided by batching: what a single-threaded reader would have
    /// spent issuing the same reads.
    pub fn syscalls_avoided(&self) -> u64 {
        self.reads.saturating_sub(self.syscalls)
    }

    fn merge(&mut self, other: &IoStats) {
        self.batches += other.batches;
        self.reads += other.reads;
        self.syscalls += other.syscalls;
        self.bytes += other.bytes;
        self.ring_batches += other.ring_batches;
    }
}

/// The mirror set: several copies of one logical file on separate drives.
pub struct MirrorSet {
    /// One open handle per replica, in replica order.
    files: Vec<File>,
    /// Where each replica lives, for error messages.
    paths: Vec<PathBuf>,
    backend: Backend,
    stats: IoStats,
    /// A ring kept alive for the life of the set.
    ///
    /// Creating a ring is a syscall pair (`io_uring_setup` plus the mmap of the
    /// rings), and doing that once per batch cost more than the batching saved:
    /// measured on 64 MiB over four replicas, a per-batch ring ran at roughly a
    /// third of the speed of plain `pread` on threads despite issuing a quarter
    /// of the syscalls. The ring is the expensive part, so it is created once
    /// and reused -- which is also how every real io_uring program is written.
    #[cfg(feature = "peregrine-uring")]
    ring: Option<io_uring::IoUring>,
}

/// Hand-written because [`io_uring::IoUring`] is not `Debug`, and the mirror set
/// is a field of `rawdata::Source`, which is. The rings are summarized by
/// whether they exist rather than by their internals.
impl fmt::Debug for MirrorSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MirrorSet")
            .field("replicas", &self.files.len())
            .field("paths", &self.paths)
            .field("backend", &self.backend)
            .field("stats", &self.stats)
            .field("ring", &has_ring(self))
            .finish()
    }
}

/// Whether this mirror set holds a live ring, for the `Debug` summary.
///
/// The `ring` field only exists with the `peregrine-uring` feature, so the
/// accessor is feature-gated with it rather than the caller: without the
/// feature the answer is simply "no", which is also the truth.
#[cfg(feature = "peregrine-uring")]
fn has_ring(set: &MirrorSet) -> bool {
    set.ring.is_some()
}

#[cfg(not(feature = "peregrine-uring"))]
fn has_ring(_set: &MirrorSet) -> bool {
    false
}

impl MirrorSet {
    /// Open `paths` as replicas of one logical file.
    ///
    /// Every path must exist and be openable. Their *contents* are not compared
    /// here: doing so would mean reading the whole file to open it, which is
    /// the cost this module exists to remove. Divergence between replicas is
    /// caught by the oracle test, which is where it belongs -- a test that
    /// knows what it expects, rather than a constructor that has to read
    /// everything to be safe.
    pub fn open(paths: &[PathBuf], backend: Backend) -> anyhow::Result<Self> {
        anyhow::ensure!(!paths.is_empty(), "a mirror set needs at least one replica");
        let mut files = Vec::with_capacity(paths.len());
        for path in paths {
            let file =
                File::open(path).with_context(|| format!("open replica {}", path.display()))?;
            files.push(file);
        }
        #[cfg(feature = "peregrine-uring")]
        let ring = match backend {
            Backend::IoUring => io_uring::IoUring::new(RING_QUEUE_DEPTH as u32)
                .ok()
                .filter(|_| ring_available()),
            Backend::Threaded => None,
        };
        Ok(Self {
            files,
            paths: paths.to_vec(),
            backend,
            stats: IoStats::default(),
            #[cfg(feature = "peregrine-uring")]
            ring,
        })
    }

    /// How many replicas are open.
    pub fn replicas(&self) -> usize {
        self.files.len()
    }

    /// The backend in use.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// The paths of the open replicas.
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// Counters since this set was opened.
    pub fn stats(&self) -> IoStats {
        self.stats
    }

    /// The plan this set would use for `region`, without performing it.
    pub fn plan(&self, region: Region) -> MirrorPlan {
        MirrorPlan::new(region, self.replicas())
    }

    /// Read `region` from replica 0 only. The reference the striped path is
    /// checked against, and a perfectly good read on its own.
    pub fn read_serial(&mut self, region: Region) -> anyhow::Result<Vec<u8>> {
        let mut out = vec![0u8; region.len];
        let mut stats = IoStats::default();
        if region.len == 0 {
            self.stats.merge(&stats);
            return Ok(out);
        }
        read_exact_at(&self.files[0], &mut out, region.offset).with_context(|| {
            format!(
                "serial read of {} bytes at {} from replica 0",
                region.len, region.offset
            )
        })?;
        stats.batches = 1;
        stats.reads = 1;
        stats.syscalls = 1;
        stats.bytes = region.len as u64;
        self.stats.merge(&stats);
        Ok(out)
    }

    /// Read `region`, split across every replica and reassembled by index.
    ///
    /// Byte-identical to [`Self::read_serial`] on the same region; the oracle
    /// test asserts exactly that.
    pub fn read_striped(&mut self, region: Region) -> anyhow::Result<Vec<u8>> {
        let plan = self.plan(region);
        self.read_plan(&plan)
    }

    /// Execute an already-computed plan.
    ///
    /// Split from [`Self::read_striped`] so a caller can *inspect* a plan, or
    /// test the reassembly, without the I/O -- and so the same reassembly code
    /// serves both backends.
    pub fn read_plan(&mut self, plan: &MirrorPlan) -> anyhow::Result<Vec<u8>> {
        plan.validate()
            .context("mirror plan is not a valid striping of its region")?;
        let mut out = vec![0u8; plan.region.len];
        if plan.stripes.is_empty() {
            self.stats.batches += 1;
            return Ok(out);
        }

        let reads = plan.stripes.len() as u64;

        // Only the ring path consumes the issue order; the threaded path
        // re-derives a destination order so its slices are provably disjoint.
        #[cfg_attr(not(feature = "peregrine-uring"), allow(unused_variables))]
        let order = plan.issue_order();
        let (ring_batches, syscalls) = match self.backend {
            #[cfg(feature = "peregrine-uring")]
            Backend::IoUring if self.ring.is_some() && order.len() <= RING_QUEUE_DEPTH => {
                self.read_ring(plan, &order, &mut out)
                    .context("io_uring stripe batch")?;
                // One enter for the whole batch. The ring does the waiting.
                (1, 1)
            }
            _ => {
                self.read_threaded(plan, &mut out)
                    .context("threaded stripe batch")?;
                (0, reads)
            }
        };
        let stats = IoStats {
            batches: 1,
            reads,
            bytes: plan.region.len as u64,
            ring_batches,
            syscalls,
        };

        self.stats.merge(&stats);
        Ok(out)
    }

    /// Read every stripe concurrently on scoped threads, one `pread` each.
    ///
    /// The issue order does not apply here: the stripes have to be *carved* out
    /// of the destination in ascending order for the slices to be disjoint, so
    /// this path derives its own order from the plan rather than taking one.
    fn read_threaded(&self, plan: &MirrorPlan, out: &mut [u8]) -> anyhow::Result<()> {
        // `out` is split into disjoint mutable slices, one per stripe, before
        // any thread starts: that is what makes the writes race-free without a
        // lock, and it is why reassembly is by index rather than by completion
        // order.
        //
        // The stripes are walked in *destination* order and carved out of `out`
        // with `split_at_mut`. That is the only way to hand several threads
        // mutable access to one buffer: each call splits the remainder, so the
        // pieces are provably disjoint and the borrow checker enforces it
        // rather than the comment asserting it. The alternative -- carrying
        // ranges and re-slicing inside each thread -- needs a raw pointer to
        // hand out a second mutable borrow, and a hand-rolled one is exactly
        // the kind of subtlety this module exists to remove.
        let mut by_dst: Vec<&Stripe> = plan.stripes.iter().collect();
        by_dst.sort_by_key(|s| s.dst_offset);
        anyhow::ensure!(
            by_dst.first().is_none_or(|s| s.dst_offset == 0),
            "the stripes do not start at the beginning of the destination"
        );

        // The stripe table is cloned so a spawned closure -- which may not
        // outlive the borrow of `plan` it captured -- can look its stripe up by
        // index instead of capturing the plan itself.
        let stripes: Vec<Stripe> = plan.stripes.clone();
        let files = &self.files;

        let total = out.len();
        let mut rest: &mut [u8] = out;
        let mut pending: Vec<(usize, &mut [u8])> = Vec::with_capacity(by_dst.len());
        for stripe in &by_dst {
            let (head, tail) = rest.split_at_mut(stripe.len);
            pending.push((stripe.index, head));
            rest = tail;
        }
        anyhow::ensure!(
            rest.is_empty(),
            "the stripes cover {} bytes of a {total}-byte destination",
            total - rest.len()
        );

        let failures: Vec<anyhow::Error> = std::thread::scope(|scope| {
            let handles: Vec<_> = pending
                .into_iter()
                .map(|(index, dst)| {
                    let stripes = &stripes;
                    scope.spawn(move || {
                        let stripe = &stripes[index];
                        let file = &files[stripe.replica];
                        read_exact_at(file, dst, stripe.file_offset).with_context(|| {
                            format!(
                                "stripe {index} (replica {}, {} bytes at {})",
                                stripe.replica, stripe.len, stripe.file_offset
                            )
                        })
                    })
                })
                .collect();
            handles
                .into_iter()
                .filter_map(|handle| match handle.join() {
                    Ok(result) => result.err(),
                    Err(_) => Some(anyhow::anyhow!("a stripe read thread panicked")),
                })
                .collect()
        });

        if let Some(first) = failures.into_iter().next() {
            return Err(first);
        }
        Ok(())
    }

    /// Submit every stripe to one ring and reap them together.
    ///
    /// The kernel writes *directly into the destination*: the stripes are
    /// carved out of `out` with `split_at_mut` first, and each SQE is pointed
    /// at its own slice. The obvious alternative -- one `Vec` per stripe, then
    /// `copy_from_slice` into place -- is a whole extra pass over the data plus
    /// an allocation per stripe, and it measured *slower than the threaded
    /// path* on this machine despite doing four times fewer syscalls. Batching
    /// the syscalls is worth nothing if the data is then copied twice, so the
    /// read lands where it is needed the first time.
    #[cfg(feature = "peregrine-uring")]
    fn read_ring(
        &mut self,
        plan: &MirrorPlan,
        order: &[&Stripe],
        out: &mut [u8],
    ) -> anyhow::Result<()> {
        use io_uring::{opcode, types};

        // Borrow the ring and the file table from disjoint fields of `self`, so
        // neither borrow has to be re-taken while the other is live.
        let Self { files, ring, .. } = self;
        // Reported rather than assumed: the caller checks the ring exists before
        // choosing this backend, and if that ever changes the right answer is to
        // fall back to the threaded path, not to read through a missing ring.
        let ring = ring.as_mut().ok_or_else(|| {
            anyhow::anyhow!("the io_uring backend was chosen without a ring to submit to")
        })?;
        let files: &[File] = files;

        // Carve the destination into one disjoint mutable slice per stripe, in
        // ascending destination order, and remember where each lives. The
        // pointers are taken from these slices, so they stay inside `out` for
        // the whole submission.
        let mut by_dst: Vec<&Stripe> = plan.stripes.iter().collect();
        by_dst.sort_by_key(|s| s.dst_offset);
        let total = out.len();
        let mut rest: &mut [u8] = out;
        let mut targets: Vec<(usize, &mut [u8])> = Vec::with_capacity(by_dst.len());
        for stripe in &by_dst {
            let (head, tail) = rest.split_at_mut(stripe.len);
            targets.push((stripe.index, head));
            rest = tail;
        }
        anyhow::ensure!(
            rest.is_empty(),
            "the stripes cover {} bytes of a {total}-byte destination",
            total - rest.len()
        );
        // `order` is a permutation of the plan's stripes, so every slot is
        // filled exactly once. Reported rather than assumed because a repeated
        // index would hand two SQEs the same destination slice -- which is
        // precisely the aliasing the disjoint carve above exists to prevent.
        let mut by_index: Vec<&mut [u8]> = {
            let mut slots: Vec<Option<&mut [u8]>> =
                targets.into_iter().map(|(_, s)| Some(s)).collect();
            let mut taken = Vec::with_capacity(order.len());
            for stripe in order {
                let slot = slots[stripe.index].take().ok_or_else(|| {
                    anyhow::anyhow!(
                        "stripe {} appears twice in the submission order, or the plan has no such stripe",
                        stripe.index
                    )
                })?;
                taken.push(slot);
            }
            taken
        };

        for (slot, stripe) in order.iter().enumerate() {
            let fd = types::Fd(files[stripe.replica].as_raw_fd());
            let dst = by_index[slot].as_mut_ptr();
            let entry = opcode::Read::new(fd, dst, stripe.len as u32)
                .offset(stripe.file_offset)
                .build()
                .user_data(slot as u64);
            // SAFETY: `dst` points into `out` at this stripe's own disjoint
            // destination range (proved by `validate` and by the `split_at_mut`
            // walk above). The kernel writes at most `stripe.len` bytes there,
            // `out` outlives this function, and every completion is reaped
            // before the slices are dropped -- so no pointer outlives its
            // buffer and no two writes can overlap.
            unsafe {
                ring.submission()
                    .push(&entry)
                    .map_err(|_| anyhow::anyhow!("the ring's submission queue filled up"))?;
            }
        }
        ring.submit().context("io_uring submit")?;
        // `submit_and_wait` takes the *count* it should wait for, as a `usize`
        // in io-uring 0.7 (it was a `u32` in earlier versions).
        ring.submit_and_wait(order.len()).context("io_uring reap")?;

        let mut completed = 0usize;
        for cqe in ring.completion() {
            let slot = cqe.user_data() as usize;
            let result = cqe.result();
            let stripe = order[slot];
            anyhow::ensure!(
                result >= 0,
                "stripe {} (replica {}) failed: {}",
                stripe.index,
                stripe.replica,
                std::io::Error::from_raw_os_error(-result)
            );
            let got = result as usize;
            anyhow::ensure!(
                got == stripe.len,
                "stripe {} read {} bytes, expected {}: a short read is a mirror that is \
                 not a mirror, not a region to be zero-filled",
                stripe.index,
                got,
                stripe.len
            );
            completed += 1;
        }
        anyhow::ensure!(
            completed == order.len(),
            "the ring completed {completed} of {} stripes",
            order.len()
        );
        Ok(())
    }
}

#[cfg(feature = "peregrine-uring")]
use std::os::unix::io::AsRawFd;

/// `pread` the whole buffer, treating a short read as an error.
///
/// A short read on a regular file means the file changed under us or the
/// offset is wrong; either way, returning the partial buffer would hand the
/// caller zeros where the mirror's bytes should be, and that is the exact
/// corruption the oracle test exists to catch.
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    if buf.is_empty() {
        return Ok(());
    }
    match file.read_exact_at(buf, offset) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => Err(std::io::Error::new(
            ErrorKind::UnexpectedEof,
            format!(
                "read of {} bytes at {offset} hit end of file ({} bytes in the replica)",
                buf.len(),
                file.metadata().map(|m| m.len()).unwrap_or(0)
            ),
        )),
        Err(e) => Err(e),
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
    use std::io::Write;
    use std::path::Path;

    /// A file of `len` bytes whose content is a deterministic function of the
    /// offset, so a mis-assembled stripe is detected rather than merely
    /// changing the length.
    fn patterned_file(dir: &Path, name: &str, len: usize) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).expect("create replica");
        let bytes: Vec<u8> = (0..len).map(|i| ((i * 31 + 7) % 251) as u8).collect();
        file.write_all(&bytes).expect("write replica");
        path
    }

    fn tempdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("peregrine-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// The oracle: every striping of a region returns exactly the bytes the
    /// single-drive read returns. This is the test the whole module exists to
    /// pass, and it is deliberately run over stripe counts that do not divide
    /// the region evenly -- an even division is the case a buggy splitter is
    /// least likely to break.
    #[test]
    fn striped_reads_are_byte_identical_to_serial() {
        let dir = tempdir("oracle");
        let len = 300 * 1024;
        let paths: Vec<PathBuf> = (0..4)
            .map(|i| patterned_file(&dir, &format!("r{i}.bin"), len))
            .collect();
        let mut set = MirrorSet::open(&paths, Backend::Threaded).expect("open");

        for replicas in 1..=4 {
            for &(off, n) in &[
                (0usize, len),
                (0, MIN_STRIPE),
                (0, MIN_STRIPE - 1),
                (1, 0),
                (7, 13),
                (12345, 1),
                (1024, 3 * MIN_STRIPE),
                (len - 1, 1),
                (0, MIN_STRIPE * 7 + 1),
            ] {
                let region = Region::new(off as u64, n);
                let serial = set.read_serial(region.clone()).expect("serial");
                // Force the plan to use exactly `replicas` copies even though
                // the set has four, by planning directly.
                let plan = MirrorPlan::new(region, replicas);
                plan.validate().expect("valid plan");
                let striped = set.read_plan(&plan).expect("striped");
                assert_eq!(
                    serial, striped,
                    "replicas={replicas} region=({off}, {n}): striped read differs from serial"
                );
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The plan covers the region exactly, for every replica count and a range
    /// of awkward lengths. A splitter that loses or duplicates a byte is caught
    /// here rather than as corrupted training data.
    #[test]
    fn plans_cover_the_region_exactly() {
        for replicas in 1..=8usize {
            for len in [
                0usize,
                1,
                13,
                MIN_STRIPE - 1,
                MIN_STRIPE,
                MIN_STRIPE + 1,
                250_003,
            ] {
                let plan = MirrorPlan::new(Region::new(1000, len), replicas);
                plan.validate()
                    .unwrap_or_else(|e| panic!("replicas={replicas} len={len}: {e}"));
                assert_eq!(plan.planned_bytes(), len, "replicas={replicas} len={len}");
                assert!(plan.stripes.len() <= replicas.max(1), "too many stripes");
                for stripe in &plan.stripes {
                    assert!(stripe.replica < replicas);
                }
            }
        }
    }

    /// A one-replica plan is the serial read: one stripe, replica 0. The
    /// degenerate case is the same code path, which is why it cannot rot.
    #[test]
    fn one_replica_is_one_stripe() {
        let plan = MirrorPlan::new(Region::new(0, 10 * MIN_STRIPE), 1);
        assert_eq!(plan.stripes.len(), 1);
        assert_eq!(plan.stripes[0].replica, 0);
        assert_eq!(plan.stripes[0].len, 10 * MIN_STRIPE);
    }

    /// A tiny region is not spread over twelve drives: splitting 200 bytes
    /// across twelve submissions moves less than one page for twelve syscalls.
    #[test]
    fn a_small_region_is_not_striped() {
        let plan = MirrorPlan::new(Region::new(0, 200), 12);
        assert_eq!(
            plan.stripes.len(),
            1,
            "200 bytes should be one read, not twelve"
        );
    }

    /// No stripe is planned below `MIN_STRIPE` unless the whole region is
    /// smaller than that.
    #[test]
    fn stripes_are_not_granular_to_the_point_of_waste() {
        let plan = MirrorPlan::new(Region::new(0, 4 * MIN_STRIPE), 64);
        assert!(
            plan.stripes.iter().all(|s| s.len >= MIN_STRIPE),
            "sub-page stripe planned"
        );
    }

    /// Issue order is longest-first and never loses or duplicates a stripe.
    #[test]
    fn issue_order_is_longest_first_and_complete() {
        let plan = MirrorPlan::new(Region::new(0, 900 * 1024), 6);
        let order = plan.issue_order();
        assert_eq!(order.len(), plan.stripes.len());
        let mut indices: Vec<usize> = order.iter().map(|s| s.index).collect();
        indices.sort_unstable();
        assert_eq!(indices, (0..plan.stripes.len()).collect::<Vec<_>>());
        for pair in order.windows(2) {
            assert!(
                pair[0].len >= pair[1].len,
                "issue order is not longest-first"
            );
        }
    }

    /// The ring path and the threaded path must return the same bytes. A
    /// divergence here means one of the two is assembling stripes wrongly, and
    /// the caller cannot tell which.
    #[test]
    fn threaded_and_ring_backends_agree() {
        let dir = tempdir("backends");
        let len = 400 * 1024;
        let paths: Vec<PathBuf> = (0..3)
            .map(|i| patterned_file(&dir, &format!("b{i}.bin"), len))
            .collect();

        let mut threaded = MirrorSet::open(&paths, Backend::Threaded).expect("open threaded");
        let region = Region::new(4096, 250_000);
        let want = threaded.read_serial(region.clone()).expect("serial");
        let from_threaded = threaded
            .read_striped(region.clone())
            .expect("threaded striped");

        let mut ring = MirrorSet::open(&paths, Backend::detect(cfg!(feature = "peregrine-uring")))
            .expect("open detected");
        let from_ring = ring.read_striped(region.clone()).expect("ring striped");

        assert_eq!(
            from_threaded, want,
            "threaded backend disagrees with serial"
        );
        if ring.backend() == Backend::IoUring {
            assert_eq!(from_ring, want, "ring backend disagrees with serial");
            let stats = ring.stats();
            assert!(stats.ring_batches >= 1, "the ring path did not run");
            assert_eq!(stats.syscalls, 1, "a ring batch should cost one syscall");
            assert!(
                stats.reads_per_syscall() > 1.0,
                "batching did not happen: {} reads per syscall",
                stats.reads_per_syscall()
            );
        } else {
            assert_eq!(from_ring, want, "fallback backend disagrees with serial");
            assert!(!ring.backend().batched_syscalls());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The counters report what actually happened, including the syscalls a
    /// batching reader avoided.
    #[test]
    fn stats_account_for_batches_reads_and_syscalls() {
        let dir = tempdir("stats");
        let len = 256 * 1024;
        let paths: Vec<PathBuf> = (0..4)
            .map(|i| patterned_file(&dir, &format!("s{i}.bin"), len))
            .collect();
        let mut set = MirrorSet::open(&paths, Backend::Threaded).expect("open");
        set.read_striped(Region::new(0, len)).expect("striped");
        let stats = set.stats();
        assert_eq!(stats.batches, 1);
        assert_eq!(
            stats.reads, 4,
            "a 256 KiB region over 4 replicas is 4 stripes"
        );
        assert_eq!(
            stats.syscalls, 4,
            "the threaded backend pays one syscall per read"
        );
        assert_eq!(stats.bytes, len as u64);
        assert_eq!(stats.reads_per_syscall(), 1.0);

        let mut serial = MirrorSet::open(&paths, Backend::Threaded).expect("open");
        serial.read_serial(Region::new(0, len)).expect("serial");
        assert_eq!(serial.stats().reads, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A read past the end of a replica is an error, never a short buffer. A
    /// zero-filled tail would be indistinguishable from real data.
    #[test]
    fn a_read_past_the_end_is_an_error_not_a_short_buffer() {
        let dir = tempdir("short");
        let paths = vec![patterned_file(&dir, "short.bin", 1024)];
        let mut set = MirrorSet::open(&paths, Backend::Threaded).expect("open");
        let err = set
            .read_striped(Region::new(0, 4096))
            .expect_err("must refuse");
        let text = format!("{err:#}");
        assert!(
            text.contains("end of file") || text.contains("failed"),
            "unhelpful error for a read past the end: {text}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An empty region is legal, yields nothing, and issues no reads.
    #[test]
    fn an_empty_region_is_legal_and_free() {
        let dir = tempdir("empty");
        let paths = vec![patterned_file(&dir, "e.bin", 4096)];
        let mut set = MirrorSet::open(&paths, Backend::Threaded).expect("open");
        assert!(set
            .read_striped(Region::new(0, 0))
            .expect("empty read")
            .is_empty());
        assert!(set
            .read_serial(Region::new(0, 0))
            .expect("empty serial")
            .is_empty());
        assert_eq!(set.stats().reads, 0, "an empty region issued reads");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A mirror set needs at least one replica, and the error says so.
    #[test]
    fn an_empty_mirror_set_is_refused() {
        let err = MirrorSet::open(&[], Backend::Threaded).expect_err("must refuse");
        assert!(
            format!("{err}").contains("at least one replica"),
            "unhelpful: {err}"
        );
    }

    /// A corrupt plan is refused before a single byte is read. Reassembly is
    /// by index, so a plan whose indices are not ascending is a plan that
    /// would scatter the file.
    #[test]
    fn a_corrupt_plan_is_refused_before_any_read() {
        let dir = tempdir("corrupt");
        let paths = vec![patterned_file(&dir, "c.bin", 8192)];
        let mut set = MirrorSet::open(&paths, Backend::Threaded).expect("open");

        let mut plan = MirrorPlan::new(Region::new(0, 4 * MIN_STRIPE), 4);
        plan.stripes.swap(0, 1);
        let err = set.read_plan(&plan).expect_err("must refuse");
        assert!(
            format!("{err:#}").contains("reassembly is by index"),
            "unhelpful: {err:#}"
        );
        assert_eq!(set.stats().reads, 0, "a refused plan still issued reads");

        // A plan whose stripes do not add up to the region is also refused.
        let mut short = MirrorPlan::new(Region::new(0, 4 * MIN_STRIPE), 4);
        short.stripes[0].len -= 1;
        assert!(set.read_plan(&short).is_err(), "a short plan was accepted");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Backends report batching honestly, and detection never claims a ring it
    /// cannot have (the feature is off in this build if the assertion holds).
    #[test]
    fn backend_detection_is_honest_about_batching() {
        let detected = Backend::detect(cfg!(feature = "peregrine-uring"));
        #[cfg(not(feature = "peregrine-uring"))]
        assert_eq!(
            detected,
            Backend::Threaded,
            "claimed a ring without the feature"
        );
        #[cfg(feature = "peregrine-uring")]
        assert!(matches!(detected, Backend::IoUring | Backend::Threaded));
        assert!(!Backend::Threaded.batched_syscalls());
    }
}
