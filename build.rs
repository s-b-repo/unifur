//! Records the build's provenance for training-state checkpoints and
//! experiment records (roadmap Phase 28): the git revision the binary was
//! built from, whether the tree was dirty, the compiler, and the Burn version
//! from `Cargo.lock`. Everything falls back to `"unknown"` rather than
//! failing the build -- provenance is a record, not a requirement.

use std::process::Command;

/// Run a command and return its trimmed stdout; every failure mode is
/// spelled out so the build log says why a provenance field is `unknown`.
fn run(cmd: &str, args: &[&str]) -> Result<String, String> {
    let out = Command::new(cmd)
        .args(args)
        .output()
        .map_err(|err| format!("{cmd} {}: {err}", args.join(" ")))?;
    if !out.status.success() {
        return Err(format!(
            "{cmd} {} exited with {}: {}",
            args.join(" "),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        return Err(format!("{cmd} {} printed nothing", args.join(" ")));
    }
    Ok(text)
}

/// The value, or `unknown` with the reason on the build log.
fn or_unknown(what: &str, value: Result<String, String>) -> String {
    match value {
        Ok(v) => v,
        Err(err) => {
            println!("cargo:warning=build provenance: {what} is unknown: {err}");
            "unknown".into()
        }
    }
}

fn burn_version() -> Result<String, String> {
    let lock = std::fs::read_to_string("Cargo.lock").map_err(|err| format!("read Cargo.lock: {err}"))?;
    let mut lines = lock.lines();
    while let Some(line) = lines.next() {
        if line.trim() == "name = \"burn\"" {
            return lines
                .next()
                .and_then(|v| v.trim().strip_prefix("version = \"").map(|v| v.trim_end_matches('"').to_string()))
                .ok_or_else(|| "the burn entry in Cargo.lock has no version line".to_string());
        }
    }
    Err("Cargo.lock has no entry for burn".into())
}

fn main() {
    let revision = or_unknown("git revision", run("git", &["rev-parse", "HEAD"]));
    // A non-empty porcelain status means the tree is dirty; an empty one is
    // reported as an error by `run` and means clean.
    let dirty = match run("git", &["status", "--porcelain"]) {
        Ok(_) => true,
        Err(err) if err.contains("printed nothing") => false,
        Err(err) => {
            println!("cargo:warning=build provenance: dirty state is unknown: {err}");
            false
        }
    };
    let revision = if revision != "unknown" && dirty { format!("{revision}-dirty") } else { revision };
    println!("cargo:rustc-env=DBLOCKS_GIT_REVISION={revision}");

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rustc_version = or_unknown("rustc version", run(&rustc, &["--version"]));
    println!("cargo:rustc-env=DBLOCKS_RUSTC_VERSION={rustc_version}");

    let burn = or_unknown("burn version", burn_version());
    println!("cargo:rustc-env=DBLOCKS_BURN_VERSION={burn}");

    println!("cargo:rerun-if-changed=Cargo.lock");
    // In a linked worktree `.git` is a file pointing at the real git dir.
    let git_dir = match std::fs::read_to_string(".git") {
        Ok(pointer) => pointer.trim().strip_prefix("gitdir: ").map(str::to_string).unwrap_or_else(|| ".git".into()),
        Err(_) => ".git".into(),
    };
    println!("cargo:rerun-if-changed={git_dir}/HEAD");
    match std::fs::read_to_string(format!("{git_dir}/HEAD")) {
        Ok(head) => {
            if let Some(reference) = head.trim().strip_prefix("ref: ") {
                println!("cargo:rerun-if-changed={git_dir}/{reference}");
            }
        }
        Err(err) => println!("cargo:warning=build provenance: .git/HEAD unreadable, rebuilds will not track commits: {err}"),
    }
}
