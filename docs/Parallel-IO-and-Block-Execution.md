# Parallel I/O and block execution

Two subsystems added together because they answer the same question from
opposite ends: *how do we stop waiting, and how do we prove the fast path is
the same program as the slow one?*

- `peregrine` — mirror-striped reads split across every drive, batched through
  io_uring where the kernel allows it, checked byte-for-byte against a
  single-drive read.
- `blockexec` — the `sync` / `mt` / `par` selector for a batch of independent
  block computations, on a persistent pool.

---

## The I/O engine

### Why striping, and why depth

A streaming training loop is not compute-bound, it is *blocked on storage*.
A single-threaded reader issues read #1, waits, issues read #2, waits again —
the device is idle while the reader assembles results, and the reader is idle
while the device works. Two fixes follow:

- **depth**: submit N reads before waiting for any. A reader that waits after
  every request pins the device at queue depth 1 and throws away most of its
  bandwidth.
- **width**: when the same bytes live on several drives, one drive is a queue
  of depth 1 no matter how deep the submissions are. Splitting a region across
  N drives turns N sequential latencies into one.

### The rule that keeps it honest

Parallel reads are only a win if they return *the same bytes*. A striped read
that is off by one sector, or that assembles stripes in completion order, is
faster and wrong — and wrong I/O in a training loop is not a crash, it is a
model that trains on corrupted data and reports a plausible loss. So:

- stripes are assembled **by index, never by completion order**;
- every stripe read is bounds-checked, and a short read is an **error**, never a
  zero-filled tail;
- `MirrorPlan::validate` proves the stripes tile the region before a single
  syscall is issued;
- `read_striped` is checked byte-for-byte against `read_serial` by the oracle
  test, on every stripe count including the awkward ones.

### Measured, not asserted

`dblocks io --path A --mirror B --mirror C --mirror D` and
`examples/perf_probe.rs` measure all three paths. On one 12-core machine, four
replicas that are four files in a *single* directory (32 MiB each):

| path | throughput | reads | syscalls | reads/syscall |
|---|---|---|---|---|
| serial (1 drive) | 851–1003 MiB/s | 1 | 1 | 1.00 |
| striped (`pread` on threads) | 2063–2562 MiB/s | 4 | 4 | 1.00 |
| striped (io_uring, persistent) | 897–986 MiB/s | 4 | 1 | **4.00** |

All three returned identical bytes, on every repeat.

Three things that table says, none of them flattering to the usual pitch:

1. **Striping is the win: ~2.7x.** That is the effect, and it is immediate.
2. **The ring is not a throughput win *here*.** It does exactly what it
   promises — four reads per `io_uring_enter`, visible in the counters — but
   saving three `enter` calls out of a 32 MiB transfer is noise beside the
   memory traffic. The threaded path moves the same bytes with four cores; the
   ring moves them with the kernel's async workers. On a *cold* cache over a
   *slow* device, where each read is latency-bound rather than
   bandwidth-bound, the balance would move. This fixture cannot show that, and
   the docs say so rather than implying a number it did not earn.
3. **These replicas share one device**, which is the configuration where
   striping helps least and the one every developer has by default. The payoff
   case is genuinely separate drives.

Two implementation details came from the measurements, not from theory:

- the kernel writes **directly into the destination**. An earlier version read
  into a per-stripe `Vec` and copied — an extra full pass over the data, which
  cost roughly half the throughput (507 → 964 MiB/s once removed);
- the ring is created **once per `MirrorSet`** and reused. Creating one per
  batch costs a syscall pair and an `mmap` each time, and that is what made the
  first version slower still.

### Backends and the fallback

| backend | mechanism | syscalls |
|---|---|---|
| `IoUring` | one `io_uring_enter` per batch | per batch |
| `Threaded` | `pread` on a scoped thread pool | per read |

Both run the *same* plan, the *same* striping and the *same* ordering
guarantee; only the syscall count differs. A ring that cannot be created (old
kernel, seccomp, no permission) degrades automatically — `Backend::detect`
probes at construction rather than at first use, so a run never discovers the
fallback halfway through, after it has already reported timings.

The feature is off by default. `peregrine-uring` turns the fast path on.

### Wired into the dataset

`rawdata::MirroredSplit` reads the existing fixed-record format through the
engine, and `RawImageDataset::mirrored` selects it:

```rust
let mut ds = RawImageDataset::mirrored(
    &paths, format, batch_size, mean, std, Backend::detect(true),
)?;
```

A batch of records is *already* one contiguous region after the indices are
sorted, which is exactly the shape striping wants — so the existing run
coalescing and the striping compose rather than compete. Coalescing finds the
runs; striping splits each run across the drives.

`MirrorSet::open` deliberately does **not** verify replicas against each other.
Doing so would mean reading the whole file to open anything, which is the cost
the module exists to remove. Divergence between replicas is caught by the
oracle test instead, which knows what it expects.

---

## Block execution modes

A batch of independent work items — several windows, micro-batches, spans, or
candidate constructions — is internally sequential but mutually independent.
There are exactly three honest ways to run it:

| mode | mechanism | pay per | use when |
|---|---|---|---|
| `sync` | one item, calling thread | — | one item already saturates the machine; or measuring a baseline |
| `mt` | one OS thread per item | batch | few, large items |
| `par` | persistent pool, machine-sized | run | thousands of batches (the default) |

`mt` and `par` are not synonyms: thread spawn is tens of microseconds against a
block forward of milliseconds, so per-batch spawning is affordable only for
large items — and a persistent pool also *bounds* the thread count, which
matters because an unbounded spawn over a 1024-item batch on a 64-core box is
1024 threads thrashing 64 cores, slower than not parallelizing at all.

### The guarantee

A parallel mode that changes the *result* is not a faster mode, it is a
different program. Two things are therefore guaranteed and tested:

1. **Order-independence.** Results are collected into a pre-sized slot per item
   and returned in *submission* order regardless of completion order. A mode
   that returned them in completion order would make any order-dependent
   computation — running statistics, a stateful RNG, an accumulating loss —
   non-deterministic, and a run that is not reproducible cannot be compared to
   its own baseline.
2. **Error, not panic.** A failing or panicking worker is captured and
   re-raised in the caller's thread, naming the item index. A mode that
   swallowed a failing item would silently train on fewer batches than the run
   claims.

`BlockExecMode::verify_equivalent` states the contract as a function, so it can
be called from the test suite *and* by a run that wants to prove its own
configuration before committing hours to it.

### Wiring

- `checkpoint::load_all(modules, paths, device, mode)` loads a batch of
  checkpoints in the chosen mode. Results are in input order; a module/path
  count mismatch is refused rather than paired by index, because a permuted
  result would pair a model with the wrong weights and nothing downstream would
  notice.
- `dblocks io --block-modes` compares the three modes on uneven work and checks
  each against the serial expectation.
- Modes serialize as `sync` / `mt` / `par` — the same names the CLI and the
  metrics use, so an experiment record can be read back and pasted into a
  command line.

---

## What is not claimed

- **The ring is not faster here**, and the docs say so.
- **Striping needs genuinely separate devices** to pay. Two paths on one disk
  is the same device queue asked twice.
- **The cold-cache, slow-device case is untested** — it needs storage this
  machine does not have. The mechanism is right for it; the number is not
  measured.
- `load_all` parallelizes *file loading*. It does not parallelize the
  transformer layers inside a block, which are sequential by construction (each
  depends on the last).
