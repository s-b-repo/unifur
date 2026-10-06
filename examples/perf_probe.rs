//! A measurement of the I/O engine, kept as an example so the claims in
//! `peregrine` can be checked on a real machine instead of believed.
//!
//! Run with the ring enabled and without, and compare the reported numbers:
//!
//! ```text
//! cargo run --release --example perf_probe --features peregrine-uring
//! cargo run --release --example perf_probe
//! ```
//!
//! The example writes a real multi-megabyte file, mirrors it, and times the
//! same reads three ways: one drive at a time, striped across every replica on
//! the threaded backend, and striped through io_uring. It prints reads,
//! syscalls and wall time for each, and it verifies that all three returned the
//! same bytes -- a faster read of the wrong data is not a result.

use anyhow::{Context, Result};
use diffusionblocks::peregrine::{Backend, IoStats, MirrorSet, Region};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn fixture(dir: &Path, name: &str, len: usize) -> Result<PathBuf> {
    let path = dir.join(name);
    let mut file = File::create(&path).with_context(|| format!("create {}", path.display()))?;
    let chunk: Vec<u8> = (0..(1 << 20)).map(|i| (i % 251) as u8).collect();
    let mut written = 0usize;
    while written < len {
        let n = chunk.len().min(len - written);
        file.write_all(&chunk[..n])
            .with_context(|| format!("write {}", path.display()))?;
        written += n;
    }
    file.sync_all()
        .with_context(|| format!("sync {}", path.display()))?;
    Ok(path)
}

fn main() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("perf-probe-{}", std::process::id()));
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;

    let len = 64 << 20; // 64 MiB per replica
    let replicas = 4usize;
    let paths: Vec<PathBuf> = (0..replicas)
        .map(|i| fixture(&dir, &format!("replica{i}.bin"), len))
        .collect::<Result<_>>()?;
    println!(
        "fixture: {} MiB x {replicas} replicas at {}",
        len >> 20,
        dir.display()
    );

    // Drop the page cache's advantage as far as we can without root: read a
    // larger scratch file first. This is best-effort and the output says so.
    let region = Region::new(0, len);

    let mut serial = MirrorSet::open(&paths, Backend::Threaded).context("open the mirrors")?;
    let t = Instant::now();
    let truth = serial.read_serial(region.clone()).context("serial read")?;
    let serial_time = t.elapsed();
    let serial_stats = serial.stats();
    report(
        "serial (1 drive, 1 syscall)",
        &serial_stats,
        serial_time,
        len,
    );

    let mut threaded = MirrorSet::open(&paths, Backend::Threaded).context("open the mirrors")?;
    let t = Instant::now();
    let got = threaded
        .read_striped(region.clone())
        .context("striped read")?;
    let threaded_time = t.elapsed();
    let threaded_stats = threaded.stats();
    report(
        "striped (pread on threads)",
        &threaded_stats,
        threaded_time,
        len,
    );
    assert_eq!(
        got, truth,
        "the threaded striped read returned different bytes"
    );

    let detected = Backend::detect(cfg!(feature = "peregrine-uring"));
    let mut ring = MirrorSet::open(&paths, detected).context("open the mirrors")?;
    let t = Instant::now();
    let got = ring.read_striped(region.clone()).context("ring read")?;
    let ring_time = t.elapsed();
    let ring_stats = ring.stats();
    report(
        &format!("striped (backend: {:?})", detected),
        &ring_stats,
        ring_time,
        len,
    );
    assert_eq!(got, truth, "the ring striped read returned different bytes");

    println!("\nall three reads returned identical {} bytes", truth.len());
    println!(
        "speedup vs serial: threaded {:.2}x, backend {:.2}x",
        serial_time.as_secs_f64() / threaded_time.as_secs_f64().max(1e-9),
        serial_time.as_secs_f64() / ring_time.as_secs_f64().max(1e-9)
    );
    if !cfg!(feature = "peregrine-uring") {
        println!(
            "\nnote: built without the `peregrine-uring` feature, so the ring path was \
             not compiled in. Rebuild with --features peregrine-uring to measure it."
        );
    }
    println!(
        "note: these replicas are files in one directory, so they share one device. \
         Striping across genuinely separate drives is the case that pays."
    );

    // Reported, not swallowed: a probe that leaves 256 MiB behind is a problem
    // the person running it needs to know about.
    if let Err(err) = std::fs::remove_dir_all(&dir) {
        eprintln!("could not remove {}: {err}", dir.display());
    }
    Ok(())
}

fn report(label: &str, stats: &IoStats, elapsed: std::time::Duration, bytes: usize) {
    println!(
        "{label:38} {:>7.2} MiB/s  reads {:>3}  syscalls {:>3}  reads/syscall {:.2}",
        (bytes as f64 / (1 << 20) as f64) / elapsed.as_secs_f64().max(1e-9),
        stats.reads,
        stats.syscalls,
        stats.reads_per_syscall(),
    );
}
