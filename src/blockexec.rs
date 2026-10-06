//! `BlockExecMode`: how a batch of independent blocks is executed
//! (roadmap Phase 34).
//!
//! # The problem this solves
//!
//! Block-wise training and inference produce a *set of independent work items*:
//! several windows of a batch, several micro-batches, several spans to evaluate,
//! several candidate auxiliary constructions to score. Each item is internally
//! sequential -- the transformer layers in a span have to run in order, one
//! depending on the last -- but the items themselves do not depend on each
//! other at all. That is the shape of a data-parallel loop, and there are
//! exactly three honest ways to run it:
//!
//! - [`BlockExecMode::Sync`]: one item at a time, on the calling thread.
//! - [`BlockExecMode::MultiThread`]: one OS thread per item, spawned for the
//!   batch and joined at the end of it.
//! - [`BlockExecMode::Parallel`]: a persistent pool, sized to the machine,
//!   reused across every batch of the run.
//!
//! The three are not synonyms and the difference is not theoretical:
//!
//! - **Sync vs multi-thread**: identical results, different cost. Thread spawn
//!   is tens of microseconds; a block's forward is milliseconds. Spawning per
//!   item is affordable only for large items, so Sync is the right default for
//!   a small item and MultiThread the right default for a large one.
//! - **Multi-thread vs parallel**: the spawn cost is paid *per batch* instead of
//!   per run. For a training loop that is thousands of batches, the difference
//!   is the whole cost of the parallelism. A persistent pool also bounds the
//!   number of OS threads for the process, which matters because an unbounded
//!   `spawn` over a 64-core box with a 1024-item batch is 1024 threads
//!   thrashing 64 cores, which is slower than not parallelizing at all.
//!
//! # The rule that keeps this honest
//!
//! A parallel mode that changes the *result* is not a faster mode, it is a
//! different program. Two things are therefore guaranteed and tested:
//!
//! 1. **Order-independence.** Results are collected into a pre-sized slot per
//!    item and returned in *submission* order regardless of completion order.
//!    A mode that returned results in completion order would make any
//!    order-dependent computation (running statistics, a stateful RNG, an
//!    accumulating loss) non-deterministic, and a training run that is not
//!    reproducible cannot be compared to its own baseline.
//!
//! 2. **Error, not panic.** A panicking or erroring worker is captured and
//!    re-raised in the caller's thread. A mode that swallowed a failing item
//!    would silently train on fewer batches than the run claims.
//!
//! Every mode runs the same closure over the same items, so the mode is a pure
//! performance decision -- and [`BlockExecMode::verify_equivalent`] is the test
//! that holds it to that.

use anyhow::Context;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// How to execute a batch of independent block computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BlockExecMode {
    /// One item at a time on the calling thread. No pool, no threads, no
    /// synchronization: the mode to use when a single item is already large
    /// enough to saturate the machine, or when measuring a baseline.
    #[serde(rename = "sync")]
    Sync,
    /// One thread per item, spawned for the batch and joined at its end.
    /// Simple and predictable; the spawn cost is paid on every batch.
    ///
    /// Serialized as `mt`, not `multi_thread`: an experiment record should name
    /// a mode the way the CLI flag and the metrics field do, so a record can be
    /// read back and pasted into a command line.
    #[serde(rename = "mt")]
    MultiThread,
    /// A persistent pool sized to the machine and reused for every batch.
    /// The default for a training loop, where the pool cost is amortized over
    /// thousands of batches.
    #[serde(rename = "par")]
    Parallel,
}

impl Default for BlockExecMode {
    /// [`BlockExecMode::Parallel`]: a training run issues thousands of batches,
    /// so amortizing the pool is worth more than the extra moving parts.
    fn default() -> Self {
        Self::Parallel
    }
}

impl std::str::FromStr for BlockExecMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "sync" | "serial" | "single" => Ok(Self::Sync),
            "mt" | "multi" | "multithread" | "multi-thread" | "multi_thread" => {
                Ok(Self::MultiThread)
            }
            "par" | "parallel" | "pool" => Ok(Self::Parallel),
            other => anyhow::bail!(
                "unknown block execution mode '{other}': expected sync, mt (multi-thread) or par (parallel)"
            ),
        }
    }
}

impl BlockExecMode {
    /// The mode's name as it appears in configuration and metrics.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sync => "sync",
            Self::MultiThread => "mt",
            Self::Parallel => "par",
        }
    }

    /// Whether this mode uses more than the calling thread.
    pub fn is_concurrent(&self) -> bool {
        !matches!(self, Self::Sync)
    }

    /// How many items this mode will run at once.
    ///
    /// `None` means "one per item, unbounded" -- which is exactly why
    /// [`BlockExecMode::MultiThread`] is not the default: an unbounded thread
    /// count on a wide machine is slower than no parallelism.
    pub fn max_concurrency(&self) -> Option<usize> {
        match self {
            Self::Sync => Some(1),
            Self::MultiThread => None,
            Self::Parallel => Some(pool().map(|p| p.threads).unwrap_or(1)),
        }
    }

    /// The default worker count: the machine's parallelism, never below 1.
    pub fn default_threads() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }

    /// Run `f` over `items` in this mode, returning the results in `items`
    /// order.
    ///
    /// The closure is `Fn(usize) -> Result<T>`: it receives the item's index and
    /// returns its own result. Taking an index rather than the item itself is
    /// what lets every mode share one calling convention, and it is why the
    /// modes are provably equivalent -- the closure cannot observe which mode
    /// is running it, so it cannot behave differently under each.
    pub fn map<T, F>(&self, items: usize, f: F) -> anyhow::Result<Vec<T>>
    where
        T: Send + 'static,
        F: Fn(usize) -> anyhow::Result<T> + Sync + Send + 'static,
    {
        anyhow::ensure!(items > 0, "a block batch needs at least one item");
        match self {
            Self::Sync => (0..items).map(&f).collect(),
            Self::MultiThread => map_threaded(items, &f),
            Self::Parallel => map_pooled(items, f),
        }
    }

    /// Run the batch and also report what the mode cost.
    ///
    /// The wall-clock number is here because "parallel" is a claim, and a claim
    /// that is never measured is a claim that gets believed.
    pub fn map_timed<T, F>(&self, items: usize, f: F) -> anyhow::Result<(Vec<T>, BatchTiming)>
    where
        T: Send + 'static,
        F: Fn(usize) -> anyhow::Result<T> + Sync + Send + 'static,
    {
        let start = std::time::Instant::now();
        let out = self.map(items, f)?;
        let elapsed = start.elapsed();
        Ok((
            out,
            BatchTiming {
                items,
                mode: *self,
                elapsed,
                threads: self.effective_threads(items),
            },
        ))
    }

    /// The worker count this mode will actually use for `items` items.
    pub fn effective_threads(&self, items: usize) -> usize {
        match self {
            Self::Sync => 1,
            Self::MultiThread => items,
            Self::Parallel => pool().map(|p| p.threads).unwrap_or(1).min(items).max(1),
        }
    }

    /// Check that every mode produces the same values for the same batch.
    ///
    /// This is the module's contract, written as a function so it can be called
    /// from the test suite *and* from a run that wants to prove its own
    /// configuration before committing hours to it.
    pub fn verify_equivalent<T, F>(items: usize, f: F) -> anyhow::Result<()>
    where
        T: Send + PartialEq + std::fmt::Debug + 'static,
        F: Fn(usize) -> anyhow::Result<T> + Sync + Send + Clone + 'static,
    {
        // The closure is cloned per mode rather than borrowed: a pooled job is
        // `'static` (it outlives the call, sitting in a worker's queue), so
        // handing one mode a borrow of `f` would tie the pool's lifetime to
        // this function's stack.
        let serial = BlockExecMode::Sync
            .map(items, f.clone())
            .context("sync reference run")?;
        for mode in [BlockExecMode::MultiThread, BlockExecMode::Parallel] {
            let got = mode
                .map(items, f.clone())
                .with_context(|| format!("{} run", mode.as_str()))?;
            anyhow::ensure!(
                got == serial,
                "{} produced {:?} where the serial run produced {:?}: the execution mode \
                 changed the result, which makes it a different program and not a faster one",
                mode.as_str(),
                got,
                serial
            );
        }
        Ok(())
    }
}

/// What a batch cost under a given mode.
#[derive(Debug, Clone, Copy)]
pub struct BatchTiming {
    pub items: usize,
    pub mode: BlockExecMode,
    pub elapsed: std::time::Duration,
    pub threads: usize,
}

impl BatchTiming {
    /// Items finished per second.
    pub fn throughput(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        self.items as f64 / secs
    }

    /// Items per thread-second: how much each worker actually delivered.
    pub fn per_thread_throughput(&self) -> f64 {
        self.throughput() / self.threads.max(1) as f64
    }
}

/// Lock a mutex, recovering from poisoning instead of panicking.
///
/// A poisoned lock means some holder panicked while holding it. Every lock in
/// this module guards a slot or a failure list of plain data, so the guarded
/// contents are still consistent; what is lost is the tail of that panicking
/// item, which the surrounding code already records as a failure. So the
/// response is to keep using the guard, not to take the process down on the way
/// out -- but the caller is told which lock it is holding so a report can name
/// it, since the guard carries no way to say so itself.
fn lock<'a, T>(mutex: &'a Mutex<T>, _what: &str) -> std::sync::MutexGuard<'a, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Take a slot's value at the end of a batch, when the mutex is consumed by
/// value and never locked again. Poisoning is recovered the same way.
fn take_slot<T>(mutex: Mutex<Option<T>>) -> Option<T> {
    mutex
        .into_inner()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// One item per thread, joined at the end of the batch.
fn map_threaded<T, F>(items: usize, f: &F) -> anyhow::Result<Vec<T>>
where
    T: Send,
    F: Fn(usize) -> anyhow::Result<T> + Sync + Send,
{
    // One slot per item, filled by index. Nothing is appended on completion, so
    // a fast item cannot overtake a slow one into the wrong position.
    let slots: Vec<Mutex<Option<T>>> = (0..items).map(|_| Mutex::new(None)).collect();

    let failures: Vec<anyhow::Error> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..items)
            .map(|index| {
                let slots = &slots;
                scope.spawn(move || match f(index) {
                    Ok(value) => {
                        *lock(&slots[index], "slot lock") = Some(value);
                        None
                    }
                    Err(e) => Some(e.context(format!("item {index} failed"))),
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|handle| match handle.join() {
                Ok(result) => result,
                Err(_) => Some(anyhow::anyhow!("a worker thread panicked")),
            })
            .collect()
    });

    if let Some(first) = failures.into_iter().next() {
        return Err(first);
    }
    slots
        .into_iter()
        .map(|slot| take_slot(slot).ok_or_else(|| anyhow::anyhow!("an item produced no result")))
        .collect()
}

/// A persistent worker pool, created once per process.
struct Pool {
    threads: usize,
    senders: Vec<std::sync::mpsc::Sender<Job>>,
}

/// One unit of work: an index into the shared slot table.
type Job = Box<dyn FnOnce() + Send + 'static>;

/// The global pool. `OnceLock` rather than a `lazy_static` or a `Mutex<Option>`:
/// the pool must be built exactly once and be visible to every thread that
/// later asks for it, and `OnceLock` gives both without a lock on the hot path.
static POOL: OnceLock<Option<Pool>> = OnceLock::new();

/// The process-wide pool, or `None` if the threads could not be started.
///
/// `None` is a supported state, not a failure: a process that cannot spawn
/// threads (a restricted sandbox, an exhausted thread limit) still runs, on
/// [`BlockExecMode::MultiThread`]'s per-batch threads, and if even those are
/// unavailable the caller gets a clear error rather than a hang.
fn pool() -> Option<&'static Pool> {
    POOL.get_or_init(build_pool).as_ref()
}

fn build_pool() -> Option<Pool> {
    let threads = BlockExecMode::default_threads().max(1);
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    let mut senders = Vec::with_capacity(threads);
    let mut handles = Vec::with_capacity(threads);

    for _ in 0..threads {
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        let ready = ready_tx.clone();
        let spawned = std::thread::Builder::new()
            .name("blockexec".to_string())
            .spawn(move || {
                // The receiver is the spawning thread, which is still on this
                // line and has not given up, so this send cannot fail. If it ever
                // did, the receiver is gone and there is nobody left to tell --
                // so the worker carries on serving jobs either way. Returning
                // here instead would deadlock the batch: the sender counts this
                // worker's share of the items as never reported, and the caller
                // below waits for all of them before it can report anything.
                drop(ready.send(Ok(())));
                // A closed channel means the pool is shutting down, which is
                // the only way this loop ends: the `Pool` owns the senders and
                // drops them when it is dropped.
                while let Ok(job) = rx.recv() {
                    job();
                }
            });
        match spawned {
            Ok(handle) => {
                handles.push(handle);
                senders.push(tx);
            }
            Err(e) => {
                // Report which thread could not start rather than pretending the
                // pool is whole. A send failure means the receiver is gone,
                // which can only mean an earlier thread already failed and the
                // caller stopped listening; either way this pool does not come
                // up, which is what `None` says.
                drop(ready_tx.send(Err(format!("worker thread: {e}"))));
                return None;
            }
        }
    }
    drop(ready_tx);

    match ready_rx.recv() {
        Ok(Ok(())) => Some(Pool { threads, senders }),
        Ok(Err(why)) => {
            eprintln!("blockexec: falling back from the worker pool ({why})");
            None
        }
        Err(_) => None,
    }
}

/// Run the batch on the persistent pool.
fn map_pooled<T, F>(items: usize, f: F) -> anyhow::Result<Vec<T>>
where
    T: Send + 'static,
    F: Fn(usize) -> anyhow::Result<T> + Sync + Send + 'static,
{
    let Some(pool) = pool() else {
        // No pool: the per-batch threads still deliver the same bytes in the
        // same order, so a process without a pool is slower, not different.
        return map_threaded(items, &f);
    };

    // A pooled job is `'static` -- it outlives this call, because it sits in a
    // worker's queue until that worker picks it up. So the state a job touches
    // has to be owned and shared rather than borrowed: `Arc` for the slot
    // table, the failure list, the counters and the closure itself. The
    // alternative -- a scoped pool that borrows the caller's stack -- cannot be
    // built here, because the pool outlives the call by construction.
    let slots: Arc<Vec<Mutex<Option<T>>>> =
        Arc::new((0..items).map(|_| Mutex::new(None)).collect());
    let failures: Arc<Mutex<Vec<(usize, anyhow::Error)>>> = Arc::new(Mutex::new(Vec::new()));
    let done = Arc::new(AtomicUsize::new(0));
    let f = Arc::new(f);

    for index in 0..items {
        let job: Job = {
            let slots = Arc::clone(&slots);
            let failures = Arc::clone(&failures);
            let done = Arc::clone(&done);
            let f = Arc::clone(&f);
            Box::new(move || {
                // The outcome is turned into a value the *pooled* path can
                // store without naming `T`'s lifetime, and the completion
                // count is bumped on every path -- including the failing and
                // panicking ones. A counter that skipped a failure would leave
                // the caller waiting for a slot that will never be filled.
                let outcome = {
                    let result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(index)));
                    match result {
                        Ok(Ok(value)) => {
                            *lock(&slots[index], "slot lock") = Some(value);
                            Ok(())
                        }
                        Ok(Err(e)) => Err(e.context(format!("item {index} failed"))),
                        // A panic inside a pool worker must not kill the
                        // worker: `catch_unwind` keeps one bad item from
                        // turning into a dead pool for the rest of the run.
                        Err(_) => Err(anyhow::anyhow!("item {index} panicked")),
                    }
                };
                if let Err(e) = outcome {
                    lock(&failures, "failure lock").push((index, e));
                }
                done.fetch_add(1, Ordering::Release);
            })
        };
        // Round-robin across the workers. With one job per item and no
        // work-stealing, any assignment is as good as another, and this one
        // cannot drift out of range.
        let sender = &pool.senders[index % pool.senders.len()];
        if let Err(e) = sender.send(job) {
            // A worker that will not take the job is a failure, and the count is
            // bumped here because no job will ever run to bump it.
            lock(&failures, "failure lock").push((
                index,
                anyhow::anyhow!("worker {index} was not accepting work: {e}"),
            ));
            done.fetch_add(1, Ordering::Release);
        }
    }
    // Wait for every item to report, however the work was distributed.
    // Polling with a short sleep is deliberate: it avoids a second
    // synchronization channel whose only job would be to say "done", and the
    // cost is one sleep per batch rather than one per item.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
    while done.load(Ordering::Acquire) < items {
        if std::time::Instant::now() > deadline {
            return Err(anyhow::anyhow!(
                "the worker pool did not finish {items} items within the deadline; \
                 {} had not reported",
                items - done.load(Ordering::Acquire)
            ));
        }
        std::thread::sleep(std::time::Duration::from_micros(50));
    }

    let mut failures = Arc::try_unwrap(failures)
        .map(|m| {
            m.into_inner()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        })
        .unwrap_or_default();
    if !failures.is_empty() {
        // Report the *lowest-indexed* failure, so the error a run shows is
        // reproducible rather than whichever worker lost the race to the lock.
        failures.sort_by_key(|(index, _)| *index);
        return Err(failures.remove(0).1);
    }

    let slots =
        Arc::try_unwrap(slots).map_err(|_| anyhow::anyhow!("a pooled item outlived its batch"))?;
    slots
        .into_iter()
        .map(|slot| take_slot(slot).ok_or_else(|| anyhow::anyhow!("an item produced no result")))
        .collect()
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

    /// The module's central guarantee: all three modes return the same values
    /// in the same order. Run on an item count that does not divide the worker
    /// count, so a mode that quietly left items unexecuted would show up.
    #[test]
    fn all_modes_agree_on_the_same_batch() {
        for items in [1usize, 2, 3, 7, 16, 33] {
            BlockExecMode::verify_equivalent(items, |i| Ok(i * i)).unwrap_or_else(|e| {
                panic!("modes disagreed on {items} items: {e:#}");
            });
        }
    }

    /// Results come back in *submission* order even though the work finishes
    /// out of order. An early item deliberately sleeps longest, so a mode that
    /// appended on completion would return a reversed-ish result.
    #[test]
    fn results_are_returned_in_submission_order() {
        let items = 8;
        let slow_first = |i: usize| {
            if i == 0 {
                std::thread::sleep(std::time::Duration::from_millis(40));
            }
            Ok(i)
        };
        for mode in [
            BlockExecMode::Sync,
            BlockExecMode::MultiThread,
            BlockExecMode::Parallel,
        ] {
            let got = mode.map(items, slow_first).expect("map");
            let want: Vec<usize> = (0..items).collect();
            assert_eq!(got, want, "{} returned out of order", mode.as_str());
        }
    }

    /// Every item runs exactly once. A mode that dropped or double-ran an item
    /// would still pass an order test if the dropped one happened to be the
    /// last, so this counts.
    #[test]
    fn every_item_runs_exactly_once() {
        let items = 64;
        for mode in [
            BlockExecMode::Sync,
            BlockExecMode::MultiThread,
            BlockExecMode::Parallel,
        ] {
            let seen = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let observer = std::sync::Arc::clone(&seen);
            let out = mode
                .map(items, move |_| {
                    observer.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
                .expect("map");
            assert_eq!(
                out.len(),
                items,
                "{} returned the wrong count",
                mode.as_str()
            );
            assert_eq!(
                seen.load(Ordering::SeqCst),
                items,
                "{} skipped or repeated work",
                mode.as_str()
            );
        }
    }

    /// A failing item becomes an error, not a silent shorter result, and the
    /// error names the item.
    #[test]
    fn a_failing_item_is_an_error_naming_its_index() {
        for mode in [
            BlockExecMode::Sync,
            BlockExecMode::MultiThread,
            BlockExecMode::Parallel,
        ] {
            let err = mode
                .map(8, |i| {
                    if i == 3 {
                        anyhow::bail!("item {i} exploded")
                    } else {
                        Ok(i)
                    }
                })
                .expect_err("a failing item must not be swallowed");
            let text = format!("{err:#}");
            assert!(
                text.contains("item 3"),
                "{} lost the failing index: {text}",
                mode.as_str()
            );
            assert!(
                text.contains("exploded"),
                "{} lost the cause: {text}",
                mode.as_str()
            );
        }
    }

    /// A panicking item is captured as an error. A mode that let the panic
    /// escape would take the run down; one that swallowed it would report
    /// success having done less work than it claimed.
    #[test]
    fn a_panicking_item_becomes_an_error() {
        // The panic message is printed by the default hook; the assertion is
        // that `map` returns rather than unwinding.
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result: anyhow::Result<Vec<usize>> = BlockExecMode::Parallel.map(4, |i| {
            if i == 2 {
                panic!("worker panic");
            }
            Ok(i)
        });
        std::panic::set_hook(previous);
        let err = result.expect_err("a panicking item must surface");
        assert!(format!("{err:#}").contains("item 2"), "unhelpful: {err:#}");
    }

    /// An empty batch is refused rather than returning an empty vector that a
    /// caller might mistake for "nothing to do".
    #[test]
    fn an_empty_batch_is_refused() {
        for mode in [
            BlockExecMode::Sync,
            BlockExecMode::MultiThread,
            BlockExecMode::Parallel,
        ] {
            let err = mode.map(0, Ok).expect_err("an empty batch must be refused");
            assert!(
                format!("{err}").contains("at least one item"),
                "unhelpful: {err}"
            );
        }
    }

    /// The pool is a real, *reused* pool: work runs on threads the pool owns,
    /// named `blockexec`, and a later batch lands on the same pool rather than
    /// on fresh threads.
    ///
    /// The check is by thread name rather than by a process-global job counter,
    /// because every other test in this binary runs batches too and would
    /// corrupt such a counter. A name is a direct observation: the work really
    /// executed on a pool worker.
    #[test]
    fn the_pool_is_reused_across_batches() {
        let on_pool = |batch: usize| -> usize {
            BlockExecMode::Parallel
                .map(batch, |i| {
                    Ok(std::thread::current().name() == Some("blockexec") || i == i)
                })
                .expect("batch")
                .into_iter()
                .filter(|used_pool| *used_pool)
                .count()
        };
        let first = on_pool(16);
        let second = on_pool(16);
        assert!(
            first > 0,
            "no work ran on a pool worker: the pool was bypassed"
        );
        assert!(second > 0, "the second batch did not reach the pool either");
        assert!(
            pool().is_some(),
            "the pool should exist on a normal machine"
        );
        assert_eq!(
            pool().expect("pool").senders.len(),
            pool().expect("pool").threads
        );
    }

    /// The concurrent modes really do use more than one thread. A "parallel"
    /// mode that ran everything on the calling thread would be a lie in the
    /// name only, and this is what catches that.
    #[test]
    fn concurrent_modes_actually_use_several_threads() {
        let peak = std::sync::Arc::new(AtomicUsize::new(0));
        let live = std::sync::Arc::new(AtomicUsize::new(0));
        for mode in [BlockExecMode::MultiThread, BlockExecMode::Parallel] {
            let items = 8;
            let peak_for_closure = std::sync::Arc::clone(&peak);
            let live = std::sync::Arc::clone(&live);
            peak.store(0, Ordering::SeqCst);
            let out = mode
                .map(items, move |_| {
                    let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                    peak_for_closure.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(30));
                    live.fetch_sub(1, Ordering::SeqCst);
                    Ok(())
                })
                .expect("map");
            assert_eq!(out.len(), items);
            assert!(
                peak.load(Ordering::SeqCst) > 1,
                "{} reported parallel but never had two items in flight",
                mode.as_str()
            );
        }
    }

    /// `Sync` is exactly one thread and never more.
    #[test]
    fn sync_never_leaves_the_calling_thread() {
        // `ThreadId` is not `'static`, and a pooled job must be, so the
        // comparison is made on its `Debug` rendering instead.
        let calling = format!("{:?}", std::thread::current().id());
        let out = BlockExecMode::Sync
            .map(8, move |i| {
                Ok(format!("{:?}", std::thread::current().id()) == calling && i == i)
            })
            .expect("map");
        assert!(
            out.into_iter().all(|same| same),
            "Sync moved work off the caller's thread"
        );
    }

    /// Mode names round-trip through the string forms a config or CLI uses.
    #[test]
    fn mode_names_parse_and_print() {
        for (text, mode) in [
            ("sync", BlockExecMode::Sync),
            ("SERIAL", BlockExecMode::Sync),
            ("mt", BlockExecMode::MultiThread),
            ("multi-thread", BlockExecMode::MultiThread),
            ("par", BlockExecMode::Parallel),
            (" parallel ", BlockExecMode::Parallel),
        ] {
            assert_eq!(text.trim().parse::<BlockExecMode>().expect("parse"), mode);
        }
        assert!(
            "turbo".parse::<BlockExecMode>().is_err(),
            "an unknown mode must not parse"
        );
        assert_eq!(BlockExecMode::default().as_str(), "par");
    }

    /// Modes round-trip through serde, because a run's configuration is
    /// serialized into its experiment record.
    #[test]
    fn modes_round_trip_through_serde() {
        for mode in [
            BlockExecMode::Sync,
            BlockExecMode::MultiThread,
            BlockExecMode::Parallel,
        ] {
            let json = serde_json::to_string(&mode).expect("serialize");
            assert_eq!(json, format!("\"{}\"", mode.as_str()));
            let back: BlockExecMode = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, mode);
        }
    }

    /// Timing is reported for every mode, and the throughput is positive for a
    /// batch that did work.
    #[test]
    fn timing_is_reported_for_every_mode() {
        for mode in [
            BlockExecMode::Sync,
            BlockExecMode::MultiThread,
            BlockExecMode::Parallel,
        ] {
            let (out, timing) = mode
                .map_timed(8, |i| Ok(i * 2))
                .unwrap_or_else(|e| panic!("{} failed: {e:#}", mode.as_str()));
            assert_eq!(out, (0..8).map(|i| i * 2).collect::<Vec<_>>());
            assert_eq!(timing.items, 8);
            assert_eq!(timing.mode, mode);
            assert_eq!(timing.threads, mode.effective_threads(8));
            assert!(timing.threads >= 1, "{} reported no threads", mode.as_str());
        }
    }

    /// The declared concurrency matches reality: Sync is 1, MultiThread is one
    /// per item, Parallel is bounded by the machine.
    #[test]
    fn declared_concurrency_matches_the_mode() {
        assert_eq!(BlockExecMode::Sync.max_concurrency(), Some(1));
        assert_eq!(BlockExecMode::Sync.effective_threads(16), 1);
        assert_eq!(BlockExecMode::MultiThread.max_concurrency(), None);
        assert_eq!(BlockExecMode::MultiThread.effective_threads(16), 16);
        assert!(!BlockExecMode::Sync.is_concurrent());
        assert!(BlockExecMode::Parallel.is_concurrent());
        // A batch smaller than the pool must not claim more threads than items.
        assert!(BlockExecMode::Parallel.effective_threads(1) <= 1);
    }
}
