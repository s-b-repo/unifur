//! Records the build's provenance for training-state checkpoints and
//! experiment records (roadmap Phase 28): the git revision the binary was
//! built from, whether the tree was dirty, the compiler, and the Burn version
//! from `Cargo.lock`. Everything falls back to `"unknown"` rather than
//! failing the build -- provenance is a record, not a requirement.

use std::process::Command;

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!text.is_empty()).then_some(text)
}

fn main() {
    let revision = run("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = run("git", &["status", "--porcelain"]).is_some();
    let revision = if revision != "unknown" && dirty { format!("{revision}-dirty") } else { revision };
    println!("cargo:rustc-env=DBLOCKS_GIT_REVISION={revision}");

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rustc_version = run(&rustc, &["--version"]).unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=DBLOCKS_RUSTC_VERSION={rustc_version}");

    let burn = std::fs::read_to_string("Cargo.lock")
        .ok()
        .and_then(|lock| {
            let mut lines = lock.lines();
            while let Some(line) = lines.next() {
                if line.trim() == "name = \"burn\"" {
                    return lines
                        .next()
                        .and_then(|v| v.trim().strip_prefix("version = \"").map(|v| v.trim_end_matches('"').to_string()));
                }
            }
            None
        })
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=DBLOCKS_BURN_VERSION={burn}");

    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=.git/HEAD");
    if let Ok(head) = std::fs::read_to_string(".git/HEAD") {
        if let Some(reference) = head.trim().strip_prefix("ref: ") {
            println!("cargo:rerun-if-changed=.git/{reference}");
        }
    }
}
