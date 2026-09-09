//! Content-addressed checkpointing (first slice of roadmap item 1.5).
//!
//! The filename embeds a *canonical content hash*: an ordered traversal of
//! every parameter tensor (shapes, dtypes, raw bytes) in struct-field
//! declaration order. Two models with identical weights therefore always map
//! to the same filename, even across processes.
//!
//! Why not hash the serialized file bytes? Burn records are keyed by randomly
//! generated `ParamId`s and stored in hash maps, so byte-identical weights
//! serialize to different bytes every run. The canonical hash ignores IDs and
//! container ordering while remaining sensitive to everything that affects
//! model behavior.
//!
//! Files themselves stay ordinary Burn msgpack checkpoints (loadable with
//! [`NamedMpkFileRecorder`]).

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::Context;
use burn::{
    module::{Module, ModuleVisitor, Param, Parameter},
    record::{FileRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Record, Recorder},
    tensor::{BasicOps, Bool, Int, Tensor, TensorKind, backend::Backend},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Number of hex characters kept from the sha256 digest in filenames
/// (64 bits of collision resistance, plenty for dedupe purposes).
const HASH_LEN: usize = 16;

type MpkRecorder = NamedMpkFileRecorder<FullPrecisionSettings>;

/// Disambiguates concurrent temp files within one process.
static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Serialize `module` into `dir` under a content-hashed filename.
///
/// Returns the final path. If a checkpoint with identical parameters already
/// exists, nothing is written and the existing file is reused.
pub fn save_content_addressed<B: Backend, M: Module<B>>(
    module: M,
    dir: &Path,
    stem: &str,
) -> anyhow::Result<PathBuf> {
    fs::create_dir_all(dir).with_context(|| format!("create dir {}", dir.display()))?;

    let ext = <MpkRecorder as FileRecorder<B>>::file_extension();
    let hash = canonical_hash_hex::<B, M>(&module);
    let final_path = dir.join(format!("{stem}-{}.{}", &hash[..HASH_LEN], ext));

    if !final_path.exists() {
        // The temp name is unique per process *and* per call: two concurrent
        // saves (an async save racing a synchronous one, say) must not write
        // the same path and rename each other's half-written file into place.
        let unique = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = dir.join(format!(".{stem}.{}.{unique}.tmp", std::process::id()));
        module
            .save_file(&tmp, &MpkRecorder::new())
            .map_err(|err| anyhow::anyhow!("serialize checkpoint: {err}"))?;
        // save_file appends the recorder extension via set_extension, which
        // *replaces* any existing suffix, so resolve the actual written path.
        let tmp = tmp.with_extension(ext);
        fs::rename(&tmp, &final_path)
            .with_context(|| format!("move {} to {}", tmp.display(), final_path.display()))?;
    }
    Ok(final_path)
}

/// Save on a background thread (roadmap item 13.2): the training loop keeps
/// running while serialization and I/O complete.
///
/// The module is moved into the worker thread; join the handle before
/// relying on the file existing.
pub fn save_content_addressed_async<B, M>(
    module: M,
    dir: std::path::PathBuf,
    stem: &'static str,
) -> std::thread::JoinHandle<anyhow::Result<PathBuf>>
where
    B: Backend + 'static,
    B::Device: Send + Sync + 'static,
    M: Module<B> + Send + 'static,
{
    std::thread::spawn(move || save_content_addressed(module, &dir, stem))
}

/// Load a checkpoint into `module`.
pub fn load<B, M>(module: M, path: &Path, device: &B::Device) -> anyhow::Result<M>
where
    B: Backend,
    M: Module<B>,
{
    module
        .load_file(path, &MpkRecorder::new(), device)
        .map_err(|err| anyhow::anyhow!("load {}: {err}", path.display()))
}

/// Most recently modified checkpoint in `dir` whose name starts with `stem`.
///
/// Content-addressed names carry no ordering of their own -- that is the point
/// of them -- so "latest" has to come from the filesystem's mtime.
pub fn latest_in_dir(dir: &Path, stem: &str) -> anyhow::Result<Option<PathBuf>> {
    if !dir.exists() {
        return Ok(None);
    }
    let ext = <MpkRecorder as FileRecorder<B32>>::file_extension();
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in fs::read_dir(dir).with_context(|| format!("read dir {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let matches_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with(stem) && n.ends_with(&format!(".{ext}")));
        if !matches_name {
            continue;
        }
        let modified = entry.metadata()?.modified()?;
        if best.as_ref().is_none_or(|(t, _)| modified > *t) {
            best = Some((modified, path));
        }
    }
    Ok(best.map(|(_, p)| p))
}

/// The recorder's file extension does not depend on the backend, but the
/// trait method does; this alias pins one so callers need not.
type B32 = burn::backend::NdArray<f32>;

/// sha256 over a module's parameters, hex-encoded.
///
/// Public because the expert index records a per-expert hash so an inference
/// engine can tell whether the weights it holds are the ones the manifest
/// describes.
pub fn canonical_hash_hex<B: Backend, M: Module<B>>(module: &M) -> String {
    let mut hasher = ContentHasher::default();
    module.visit(&mut hasher);
    let digest = hasher.sha.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// Feeds every visited parameter (shape | dtype | bytes, in traversal order)
/// into a sha256 state.
#[derive(Default)]
struct ContentHasher {
    sha: Sha256,
}

impl ContentHasher {
    fn hash_param<B: Backend, K: TensorKind<B> + BasicOps<B>, const D: usize>(
        &mut self,
        param: &Param<Tensor<B, D, K>>,
    ) where Tensor<B, D, K>: Parameter {
        let tensor = param.val();
        let shape = format!("{:?}", tensor.shape());
        let data = tensor.to_data();
        let dtype = format!("{:?}", data.dtype);

        self.sha.update(shape.as_bytes());
        self.sha.update([0u8]);
        self.sha.update(dtype.as_bytes());
        self.sha.update([0u8]);
        let len = data.as_bytes().len() as u64;
        self.sha.update(len.to_le_bytes());
        self.sha.update(data.as_bytes());
    }
}

impl<B: Backend> ModuleVisitor<B> for ContentHasher {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.hash_param(param);
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, Int>>) {
        self.hash_param(param);
    }

    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<B, D, Bool>>) {
        self.hash_param(param);
    }
}

// ------------------------------------------------- training state (Phase 28) --
//
// A model file alone cannot resume a run: the optimizer's moments, the
// schedules' positions, the host RNG, the EMA shadow and every host-side
// estimator the trainer keeps would all restart from scratch, and the resumed
// run would silently diverge from the one it claims to continue. The state
// lives in a directory next to the model file, named after it, so the
// content-addressed model stays exactly what it was and `latest_in_dir` still
// sees only model files.

/// Bumped when the sidecar's shape changes incompatibly.
pub const STATE_FORMAT_VERSION: u32 = 1;

/// sha256 of a file's bytes, hex-encoded, streamed in 1 MiB chunks.
pub fn file_sha256_hex(path: &Path) -> anyhow::Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut sha = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        sha.update(&buf[..n]);
    }
    Ok(hex(&sha.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// What the binary was built from, for the record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildInfo {
    /// `git rev-parse HEAD` at build time, `-dirty` if the tree had changes,
    /// `unknown` outside a repository.
    pub git_revision: String,
    pub crate_version: String,
    pub burn_version: String,
    pub rustc_version: String,
    /// `debug` or `release`.
    pub profile: String,
}

impl BuildInfo {
    pub fn current() -> Self {
        Self {
            git_revision: option_env!("DBLOCKS_GIT_REVISION").unwrap_or("unknown").to_string(),
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            burn_version: option_env!("DBLOCKS_BURN_VERSION").unwrap_or("unknown").to_string(),
            rustc_version: option_env!("DBLOCKS_RUSTC_VERSION").unwrap_or("unknown").to_string(),
            profile: if cfg!(debug_assertions) { "debug" } else { "release" }.to_string(),
        }
    }
}

/// Which data a run trained on, hashed **once** at startup.
///
/// A result that cannot say what it was trained on cannot be reproduced. The
/// hash covers every regular file under the path, in sorted order, each
/// prefixed by its name and length, so a renamed or truncated file changes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetIdentity {
    pub description: String,
    pub path: Option<String>,
    pub files: usize,
    pub bytes: u64,
    /// `None` for data with no bytes to hash (synthetic).
    pub sha256: Option<String>,
}

impl DatasetIdentity {
    /// Data generated in-process from a seed: nothing on disk to hash.
    pub fn synthetic(description: impl Into<String>) -> Self {
        Self { description: description.into(), path: None, files: 0, bytes: 0, sha256: None }
    }

    /// A file or a directory of files.
    pub fn of_path(path: &Path, description: impl Into<String>) -> anyhow::Result<Self> {
        Self::of_paths(&[path.to_path_buf()], description)
    }

    /// Several files or directories, hashed as one sorted sequence.
    pub fn of_paths(paths: &[PathBuf], description: impl Into<String>) -> anyhow::Result<Self> {
        let mut files = Vec::new();
        for root in paths {
            collect_files(root, &mut files)?;
        }
        files.sort();
        let mut sha = Sha256::new();
        let mut bytes = 0u64;
        for file in &files {
            let len = fs::metadata(file).with_context(|| format!("stat {}", file.display()))?.len();
            let name = file.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            sha.update(name.as_bytes());
            sha.update([0u8]);
            sha.update(len.to_le_bytes());
            let mut handle = fs::File::open(file).with_context(|| format!("open {}", file.display()))?;
            let mut buf = vec![0u8; 1 << 20];
            loop {
                let n = handle.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                sha.update(&buf[..n]);
            }
            bytes += len;
        }
        Ok(Self {
            description: description.into(),
            path: paths.first().map(|p| p.display().to_string()),
            files: files.len(),
            bytes,
            sha256: Some(hex(&sha.finalize())),
        })
    }

    /// Same data: same description, byte count and hash.
    pub fn matches(&self, other: &Self) -> bool {
        self.description == other.description && self.bytes == other.bytes && self.sha256 == other.sha256
    }
}

fn collect_files(root: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    let meta = fs::metadata(root).with_context(|| format!("stat {}", root.display()))?;
    if meta.is_file() {
        out.push(root.to_path_buf());
    } else if meta.is_dir() {
        for entry in fs::read_dir(root).with_context(|| format!("read dir {}", root.display()))? {
            collect_files(&entry?.path(), out)?;
        }
    }
    Ok(())
}

/// A file the state refers to, with the hash it had when the state was written.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateFile {
    /// Relative to the state directory, or absolute for the model file.
    pub path: String,
    pub sha256: String,
}

/// Everything a resumed run needs beyond the weights (roadmap Phase 28).
///
/// Written as `<model>.state/state.json` next to the content-addressed model
/// file, with the tensor-valued parts as separate records in the same
/// directory. Every file carries its sha256 so a truncated or corrupted
/// checkpoint is refused rather than resumed from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainState {
    pub format_version: u32,
    /// `dblock` or `lm`: which trainer wrote it.
    pub kind: String,
    /// Steps completed; the resumed loop starts here.
    pub step: usize,
    pub seed: u64,
    /// The host RNG, serialized whole, so the resumed run draws exactly what
    /// the uninterrupted one would have.
    pub host_rng: serde_json::Value,
    /// The training configuration, so a mismatch can be reported.
    pub config: serde_json::Value,
    pub build: BuildInfo,
    pub dataset: DatasetIdentity,
    pub model: StateFile,
    pub optimizer: Option<StateFile>,
    pub ema: Option<StateFile>,
    pub head: Option<StateFile>,
    pub head_optimizer: Option<StateFile>,
    /// Trainer-specific host state (loss scales, samplers, health, counters).
    pub extras: serde_json::Value,
    pub saved_unix_secs: u64,
}

impl TrainState {
    /// `dblocks-<hash>.mpk` -> `dblocks-<hash>.state`, a directory.
    pub fn dir_for(model_path: &Path) -> PathBuf {
        model_path.with_extension("state")
    }

    pub fn json_path(dir: &Path) -> PathBuf {
        dir.join("state.json")
    }

    /// The state next to `model_path`, if a run wrote one.
    pub fn for_model(model_path: &Path) -> anyhow::Result<Option<Self>> {
        let path = Self::json_path(&Self::dir_for(model_path));
        if !path.exists() {
            return Ok(None);
        }
        Self::read(&path).map(Some)
    }

    pub fn read(path: &Path) -> anyhow::Result<Self> {
        let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let state: Self = serde_json::from_str(&text)
            .with_context(|| format!("parse training state {}", path.display()))?;
        anyhow::ensure!(
            state.format_version == STATE_FORMAT_VERSION,
            "{} is training-state format {}, this build reads {}",
            path.display(),
            state.format_version,
            STATE_FORMAT_VERSION
        );
        Ok(state)
    }

    /// Write atomically: to a temporary name, then renamed into place.
    pub fn write(&self, dir: &Path) -> anyhow::Result<PathBuf> {
        fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        let path = Self::json_path(dir);
        let tmp = dir.join(format!(".state.{}.tmp", std::process::id()));
        let text = serde_json::to_string_pretty(self).context("serialize training state")?;
        fs::write(&tmp, format!("{text}\n")).with_context(|| format!("write {}", tmp.display()))?;
        fs::rename(&tmp, &path).with_context(|| format!("move {} to {}", tmp.display(), path.display()))?;
        Ok(path)
    }

    /// Recompute every referenced file's hash and refuse a mismatch.
    ///
    /// This is the corruption check: a checkpoint that was truncated, partly
    /// overwritten, or paired with the wrong sidecar fails here by name
    /// instead of loading into a model that is subtly wrong.
    pub fn verify_files(&self, dir: &Path) -> anyhow::Result<()> {
        let check = |file: &StateFile, what: &str| -> anyhow::Result<()> {
            let path = resolve(dir, &file.path);
            anyhow::ensure!(path.exists(), "{what} file {} is missing", path.display());
            let actual = file_sha256_hex(&path)?;
            anyhow::ensure!(
                actual == file.sha256,
                "{what} file {} is corrupted: sha256 {} does not match the recorded {}",
                path.display(),
                &actual[..16],
                &file.sha256[..16]
            );
            Ok(())
        };
        check(&self.model, "model")?;
        for (file, what) in [
            (&self.optimizer, "optimizer"),
            (&self.ema, "EMA shadow"),
            (&self.head, "uncertainty head"),
            (&self.head_optimizer, "uncertainty-head optimizer"),
        ] {
            if let Some(file) = file {
                check(file, what)?;
            }
        }
        Ok(())
    }

    /// Keys of `config` whose values differ from `other`, ignoring the keys
    /// that legitimately change between a run and its continuation.
    pub fn config_differences(&self, other: &serde_json::Value) -> Vec<String> {
        const VOLATILE: [&str; 6] = ["resume", "steps", "out_dir", "checkpoint_every", "log_file", "log_path"];
        let (Some(a), Some(b)) = (self.config.as_object(), other.as_object()) else {
            return vec!["config is not an object".into()];
        };
        let mut keys: Vec<&String> = a.keys().chain(b.keys()).collect();
        keys.sort();
        keys.dedup();
        keys.into_iter()
            .filter(|k| !VOLATILE.contains(&k.as_str()))
            .filter(|k| a.get(*k) != b.get(*k))
            .cloned()
            .collect()
    }
}

/// A path recorded in a state file, relative to the state directory unless
/// absolute.
pub fn resolve(dir: &Path, recorded: &str) -> PathBuf {
    let p = Path::new(recorded);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        dir.join(p)
    }
}

/// Save any Burn record (a module's, an optimizer's) under `dir/<name>.mpk`
/// and return its entry for the state file.
pub fn save_record<B: Backend, R: Record<B>>(record: R, dir: &Path, name: &str) -> anyhow::Result<StateFile> {
    fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let recorder = MpkRecorder::new();
    let ext = <MpkRecorder as FileRecorder<B>>::file_extension();
    // The recorder appends its own extension; give it a name without one so
    // nothing of ours gets replaced.
    let unique = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = dir.join(format!(".{name}.{}.{unique}", std::process::id()));
    <MpkRecorder as Recorder<B>>::record(&recorder, record, tmp.clone())
        .map_err(|err| anyhow::anyhow!("serialize {name}: {err}"))?;
    let tmp = tmp.with_extension(ext);
    let final_path = dir.join(format!("{name}.{ext}"));
    fs::rename(&tmp, &final_path)
        .with_context(|| format!("move {} to {}", tmp.display(), final_path.display()))?;
    Ok(StateFile { path: format!("{name}.{ext}"), sha256: file_sha256_hex(&final_path)? })
}

/// Load a record saved by [`save_record`].
pub fn load_record<B: Backend, R: Record<B>>(dir: &Path, file: &StateFile, device: &B::Device) -> anyhow::Result<R> {
    let path = resolve(dir, &file.path);
    <MpkRecorder as Recorder<B>>::load(&MpkRecorder::new(), path.clone(), device)
        .map_err(|err| anyhow::anyhow!("load {}: {err}", path.display()))
}

/// The entry for an already-written model file.
pub fn model_entry(model_path: &Path) -> anyhow::Result<StateFile> {
    Ok(StateFile { path: model_path.display().to_string(), sha256: file_sha256_hex(model_path)? })
}

/// Seconds since the Unix epoch, for the record.
pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dblock::{DblockClassifier, DblockConfig};
    use crate::vit::ViTDiTConfig;
    use burn::backend::NdArray;
    use burn::tensor::Tensor;
    use burn::tensor::backend::BackendTypes;

    type B = NdArray<f32>;
    type Device = <B as BackendTypes>::Device;

    fn tiny_vit_config() -> ViTDiTConfig {
        ViTDiTConfig::tiny(10)
    }

    fn tiny_model(device: &Device) -> DblockClassifier<B> {
        DblockClassifier::<B>::new(
            &tiny_vit_config(),
            &DblockConfig {
                num_blocks: 2,
                ..DblockConfig::default()
            },
            device,
        )
    }

    fn probe_logits(model: &DblockClassifier<B>, device: &Device) -> Tensor<B, 2> {
        let pixels = Tensor::<B, 4>::ones([2, 3, 32, 32], device);
        model.diffusion_step(pixels)
    }

    /// Unique scratch directory per test invocation.
    fn scratch_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("dblocks-test-{}-{tag}-{nanos}", std::process::id()))
    }

    #[test]
    fn test_save_produces_hashed_filename_and_dedupes() {
        let device = Default::default();
        let dir = scratch_dir("dedupe");

        // Save the SAME weights twice; fresh models would differ due to
        // random initialization.
        let model = tiny_model(&device);

        let path1 = save_content_addressed(model.clone(), &dir, "dblocks").expect("first save");
        let name1 = path1.file_name().unwrap().to_str().unwrap().to_owned();

        let path2 = save_content_addressed(model, &dir, "dblocks").expect("second save");
        assert_eq!(path1, path2);

        let entries: Vec<_> = fs::read_dir(&dir).unwrap().collect();
        assert_eq!(entries.len(), 1, "dedupe must leave exactly one file");

        // Filename shape: dblocks-<16 hex>.mpk
        assert!(name1.starts_with("dblocks-"), "{name1}");
        let hash_part = name1.trim_start_matches("dblocks-").trim_end_matches(".mpk");
        assert_eq!(hash_part.len(), HASH_LEN, "{name1}");
        assert!(hash_part.chars().all(|c| c.is_ascii_hexdigit()), "{name1}");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_different_weights_get_different_names() {
        let device = Default::default();
        let dir = scratch_dir("distinct");

        let a = tiny_model(&device);

        // Perturb one parameter so contents differ.
        let mut rec = tiny_model(&device).into_record();
        let w = rec.model.vit.embeddings.label_embeddings.weight.val().set_require_grad(false) + 1.0;
        rec.model.vit.embeddings.label_embeddings.weight =
            burn::module::Param::from_tensor(w);
        let b = tiny_model(&device).load_record(rec);

        let pa = save_content_addressed(a, &dir, "dblocks").unwrap();
        let pb = save_content_addressed(b, &dir, "dblocks").unwrap();
        assert_ne!(pa, pb, "different content must map to different files");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_latest_in_dir_picks_the_newest_matching_file() {
        let dir = scratch_dir("latest");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(latest_in_dir(&dir, "dblocks").unwrap().is_none(), "empty dir");

        std::fs::write(dir.join("dblocks-aaaa.mpk"), b"a").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(dir.join("dblocks-bbbb.mpk"), b"b").unwrap();
        // Decoys: wrong stem, wrong extension.
        std::fs::write(dir.join("other-cccc.mpk"), b"c").unwrap();
        std::fs::write(dir.join("dblocks-dddd.txt"), b"d").unwrap();

        let latest = latest_in_dir(&dir, "dblocks").unwrap().unwrap();
        assert_eq!(latest.file_name().unwrap(), "dblocks-bbbb.mpk");

        // A directory that does not exist is "no checkpoint", not an error:
        // a first run has nothing to resume from.
        assert!(latest_in_dir(&dir.join("missing"), "dblocks").unwrap().is_none());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_restores_a_saved_model() {
        let device = Default::default();
        let dir = scratch_dir("load");
        let model = tiny_model(&device);
        let before = probe_logits(&model, &device);
        let path = save_content_addressed(model, &dir, "dblocks").unwrap();

        let restored = load::<B, _>(tiny_model(&device), &path, &device).unwrap();
        let after = probe_logits(&restored, &device);
        assert_eq!((before - after).abs().max().into_scalar(), 0.0);

        // A missing file is a reported error, not a panic.
        assert!(load::<B, _>(tiny_model(&device), &dir.join("nope.mpk"), &device).is_err());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_roundtrip_preserves_outputs() {
        let device = Default::default();
        let dir = scratch_dir("roundtrip");

        let model = tiny_model(&device);
        let before = probe_logits(&model, &device);

        let path = save_content_addressed(model, &dir, "dblocks").expect("save");

        let restored = tiny_model(&device)
            .load_file(&path, &MpkRecorder::new(), &device)
            .expect("load");
        let after = probe_logits(&restored, &device);

        let diff = (before - after).abs().max().into_scalar();
        assert!(diff == 0.0, "roundtrip must be exact, max |diff| = {diff}");

        fs::remove_dir_all(&dir).unwrap();
    }
}
