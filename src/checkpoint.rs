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
use std::path::{Path, PathBuf};

use anyhow::Context;
use burn::{
    module::{Module, ModuleVisitor, Param, Parameter},
    record::{FileRecorder, FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{BasicOps, Bool, Int, Tensor, TensorKind, backend::Backend},
};
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
