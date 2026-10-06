//! Qwen3.8 weight inventory and tensor-name remap (Phase B/C of
//! `docs/Frontier-27B-Plan.md`).
//!
//! Every name below comes from HuggingFace `transformers`
//! `models/qwen3_5/modeling_qwen3_5.py` attribute paths (decoder layers expose
//! `linear_attn` *or* `self_attn`, always-plain `mlp`, `input_layernorm` +
//! `post_attention_layernorm`), **not** from vLLM internals, which rename
//! several projections (fused `in_proj_qkvz`, interleaved `ba`). Concretely
//! this model generation uses *separate* projections where Qwen3-Next fuses:
//! `in_proj_qkv` + `in_proj_z` (not `in_proj_qkvz`), `in_proj_b` +
//! `in_proj_a` (not `in_proj_ba`), and a bias-free `conv1d`. Guessing the
//! Next layout here would mis-split every linear layer, so the table keeps
//! the two generations apart on purpose.
//!
//! What this module does today: parse every checkpoint tensor name into a
//! typed slot, check its shape against the architecture dims, and audit a
//! whole shard (every weight accounted, nothing unexpected). Non-text names
//! (MTP heads, vision tower) are reported as `Foreign`, never silently
//! absorbed: the caller allow-lists the prefixes it means to skip.
//! The Burn modules these slots fill (`GatedDeltaHead`, the 64-layer trunk)
//! are later Phase C work reusing [`crate::deltanet`].

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

/// Architecture dims a remap is validated against. [`QwenArchDims::qwen38`]
/// transcribes the downloaded `config.json` (`text_config`) of the target
/// repo, checked by `qwen38_matches_downloaded_config` below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QwenArchDims {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Gated-DeltaNet key groups and value heads (`16` / `48`).
    pub linear_k_groups: usize,
    pub linear_v_heads: usize,
    pub linear_head_dim: usize,
    pub conv_kernel: usize,
    /// Every Nth layer (1-based) is full attention; the rest are linear.
    pub full_attention_interval: usize,
    /// Full-attention Q projection carries its sigmoid output gate
    /// concatenated (`q_gate` split in two downstream), doubling its width.
    pub attn_output_gate: bool,
}

impl QwenArchDims {
    pub fn qwen38() -> Self {
        Self {
            num_layers: 64,
            hidden_size: 5120,
            intermediate_size: 17408,
            vocab_size: 248320,
            num_q_heads: 24,
            num_kv_heads: 4,
            head_dim: 256,
            linear_k_groups: 16,
            linear_v_heads: 48,
            linear_head_dim: 128,
            conv_kernel: 4,
            full_attention_interval: 4,
            attn_output_gate: true,
        }
    }

    /// Value heads total (48 on Qwen3.8; stored directly, not per group).
    pub fn num_v_heads(&self) -> usize {
        self.linear_v_heads
    }

    /// Key-projection width per linear layer: `groups * dim`.
    pub fn key_dim(&self) -> usize {
        self.linear_k_groups * self.linear_head_dim
    }

    /// Value-projection width per linear layer: `v_heads * dim`.
    pub fn value_dim(&self) -> usize {
        self.linear_v_heads * self.linear_head_dim
    }

    /// `true` for 0-based `layer_idx` under the interval convention.
    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        (layer_idx + 1) % self.full_attention_interval == 0
    }
}

/// Where one checkpoint tensor belongs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QwenSlot {
    Embed,
    FinalNorm,
    LmHead,
    Layer {
        idx: usize,
        part: QwenPart,
    },
    /// Not part of the text trunk (MTP heads, vision tower, ...). Carries
    /// the raw name so the caller decides explicitly.
    Foreign(String),
}

/// One named weight inside a decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenPart {
    // Gated-DeltaNet projections (separate on this generation).
    LinearQkv,
    LinearZ,
    LinearB,
    LinearA,
    LinearConv,
    LinearConvBias,
    LinearGateNorm,
    LinearOut,
    DeltaDtBias,
    DeltaALog,
    // Full attention.
    FullQ,
    FullK,
    FullV,
    FullO,
    FullQNorm,
    FullKNorm,
    // Feed-forward (always dense on the 27B).
    MlpGate,
    MlpUp,
    MlpDown,
    InputNorm,
    PostNorm,
}

impl QwenPart {
    /// Expected full shape against `dims`, rank included, in the file's own
    /// PyTorch convention: rank-2 linears as `[out, in]` (transposed on
    /// fill into Burn, whose linears are `[in, out]`), the depthwise conv
    /// as `[channels, 1, K]`, biases and norms as rank-1.
    pub fn expected_shape(&self, dims: &QwenArchDims) -> Vec<usize> {
        let h = dims.hidden_size;
        let i = dims.intermediate_size;
        let kd = dims.key_dim();
        let vd = dims.value_dim();
        match self {
            // in_proj_qkv: [2*kd + vd, h]; z: [vd, h]; b/a: [nv, h].
            Self::LinearQkv => vec![2 * kd + vd, h],
            Self::LinearZ => vec![vd, h],
            Self::LinearB | Self::LinearA => vec![dims.linear_v_heads, h],
            Self::LinearConv => vec![kd * 2 + vd, 1, dims.conv_kernel],
            Self::LinearConvBias => vec![kd * 2 + vd],
            Self::LinearGateNorm => vec![dims.linear_head_dim],
            Self::LinearOut => vec![h, vd],
            Self::DeltaDtBias | Self::DeltaALog => vec![dims.linear_v_heads],
            Self::FullQ => {
                // The Q projection concatenates the sigmoid output gate
                // (`q_gate`, split downstream), doubling its width.
                let width = dims.num_q_heads * dims.head_dim;
                vec![
                    if dims.attn_output_gate {
                        2 * width
                    } else {
                        width
                    },
                    h,
                ]
            }
            Self::FullK | Self::FullV => vec![dims.num_kv_heads * dims.head_dim, h],
            Self::FullO => vec![h, dims.num_q_heads * dims.head_dim],
            Self::FullQNorm | Self::FullKNorm => vec![dims.head_dim],
            Self::MlpGate | Self::MlpUp => vec![i, h],
            Self::MlpDown => vec![h, i],
            Self::InputNorm | Self::PostNorm => vec![h],
        }
    }
}

/// Parse one checkpoint tensor name into a [`QwenSlot`]. `model_prefix` is
/// the text-trunk root as stored (`"model"` or `"language_model"`,
/// discovered from the first shard, never assumed here).
pub fn parse_name(name: &str, model_prefix: &str) -> Result<QwenSlot> {
    if name == format!("{model_prefix}.embed_tokens.weight") {
        return Ok(QwenSlot::Embed);
    }
    if name == format!("{model_prefix}.norm.weight") {
        return Ok(QwenSlot::FinalNorm);
    }
    if name == "lm_head.weight" {
        return Ok(QwenSlot::LmHead);
    }
    let rest = match name.strip_prefix(&format!("{model_prefix}.layers.")) {
        Some(rest) => rest,
        None => return Ok(QwenSlot::Foreign(name.to_string())),
    };
    let (idx_str, tail) = rest
        .split_once('.')
        .context(format!("malformed layer tensor {name:?}"))?;
    let idx: usize = idx_str
        .parse()
        .with_context(|| format!("bad layer index in {name:?}"))?;
    let part = match tail {
        "linear_attn.in_proj_qkv.weight" => QwenPart::LinearQkv,
        "linear_attn.in_proj_z.weight" => QwenPart::LinearZ,
        "linear_attn.in_proj_b.weight" => QwenPart::LinearB,
        "linear_attn.in_proj_a.weight" => QwenPart::LinearA,
        "linear_attn.conv1d.weight" => QwenPart::LinearConv,
        "linear_attn.conv1d.bias" => QwenPart::LinearConvBias,
        "linear_attn.norm.weight" => QwenPart::LinearGateNorm,
        "linear_attn.out_proj.weight" => QwenPart::LinearOut,
        "linear_attn.dt_bias" => QwenPart::DeltaDtBias,
        "linear_attn.A_log" => QwenPart::DeltaALog,
        "self_attn.q_proj.weight" => QwenPart::FullQ,
        "self_attn.k_proj.weight" => QwenPart::FullK,
        "self_attn.v_proj.weight" => QwenPart::FullV,
        "self_attn.o_proj.weight" => QwenPart::FullO,
        "self_attn.q_norm.weight" => QwenPart::FullQNorm,
        "self_attn.k_norm.weight" => QwenPart::FullKNorm,
        "mlp.gate_proj.weight" => QwenPart::MlpGate,
        "mlp.up_proj.weight" => QwenPart::MlpUp,
        "mlp.down_proj.weight" => QwenPart::MlpDown,
        "input_layernorm.weight" => QwenPart::InputNorm,
        "post_attention_layernorm.weight" => QwenPart::PostNorm,
        _ => return Ok(QwenSlot::Foreign(name.to_string())),
    };
    Ok(QwenSlot::Layer { idx, part })
}

/// What the audit found for one tensor.
#[derive(Debug, Clone)]
pub struct SlotReport {
    pub name: String,
    pub slot: QwenSlot,
    pub shape_ok: bool,
    pub detail: String,
}

/// Audit every tensor of a shard index: parse names, check shapes against
/// `dims`, and refuse anything unexpected. `skip_substrings` allow-lists
/// foreign names the caller means to drop (MTP heads, vision tower); a
/// foreign name outside the list fails the audit rather than vanishing.
/// `model_prefix` is tried as given; the caller discovers it from shard 1.
pub fn audit_tensors(
    entries: &[(String, Vec<usize>)],
    dims: &QwenArchDims,
    model_prefix: &str,
    skip_substrings: &[&str],
) -> Result<Vec<SlotReport>> {
    let mut reports = Vec::with_capacity(entries.len());
    for (name, shape) in entries {
        let slot = parse_name(name, model_prefix)?;
        match &slot {
            QwenSlot::Foreign(_) => {
                let allowed = skip_substrings.iter().any(|s| name.contains(s));
                ensure!(
                    allowed,
                    "unexpected foreign tensor {name:?}: not in the Qwen text layout and not allow-listed"
                );
                reports.push(SlotReport {
                    name: name.clone(),
                    slot,
                    shape_ok: true,
                    detail: "foreign, allow-listed skip".to_string(),
                });
            }
            QwenSlot::Embed => {
                reports.push(check(name, &slot, shape, dims.vocab_size, dims.hidden_size))
            }
            QwenSlot::FinalNorm => reports.push(check_rank1(name, &slot, shape, dims.hidden_size)),
            QwenSlot::LmHead => {
                reports.push(check(name, &slot, shape, dims.vocab_size, dims.hidden_size))
            }
            QwenSlot::Layer { idx, part } => {
                ensure!(
                    *idx < dims.num_layers,
                    "layer index {idx} beyond {} layers",
                    dims.num_layers
                );
                let kind_ok = if dims.is_full_attention(*idx) {
                    matches!(
                        part,
                        QwenPart::FullQ
                            | QwenPart::FullK
                            | QwenPart::FullV
                            | QwenPart::FullO
                            | QwenPart::FullQNorm
                            | QwenPart::FullKNorm
                            | QwenPart::MlpGate
                            | QwenPart::MlpUp
                            | QwenPart::MlpDown
                            | QwenPart::InputNorm
                            | QwenPart::PostNorm
                    )
                } else {
                    matches!(
                        part,
                        QwenPart::LinearQkv
                            | QwenPart::LinearZ
                            | QwenPart::LinearB
                            | QwenPart::LinearA
                            | QwenPart::LinearConv
                            | QwenPart::LinearConvBias
                            | QwenPart::LinearGateNorm
                            | QwenPart::LinearOut
                            | QwenPart::DeltaDtBias
                            | QwenPart::DeltaALog
                            | QwenPart::MlpGate
                            | QwenPart::MlpUp
                            | QwenPart::MlpDown
                            | QwenPart::InputNorm
                            | QwenPart::PostNorm
                    )
                };
                ensure!(
                    kind_ok,
                    "tensor {name:?} sits on a {} layer {idx} but belongs to the other kind",
                    if dims.is_full_attention(*idx) {
                        "full-attention"
                    } else {
                        "linear"
                    }
                );
                let want = part.expected_shape(dims);
                let ok = *shape == want;
                reports.push(SlotReport {
                    name: name.to_string(),
                    slot: slot.clone(),
                    shape_ok: ok,
                    detail: if ok {
                        "ok".to_string()
                    } else {
                        format!("shape {shape:?}, expected {want:?}")
                    },
                });
            }
        }
    }
    Ok(reports)
}

fn check(name: &str, slot: &QwenSlot, shape: &[usize], rows: usize, cols: usize) -> SlotReport {
    let ok = shape == [rows, cols];
    SlotReport {
        name: name.to_string(),
        slot: slot.clone(),
        shape_ok: ok,
        detail: if ok {
            "ok".to_string()
        } else {
            format!("shape {shape:?}, expected [{rows}, {cols}]")
        },
    }
}

fn check_rank1(name: &str, slot: &QwenSlot, shape: &[usize], len: usize) -> SlotReport {
    let ok = shape == [len];
    SlotReport {
        name: name.to_string(),
        slot: slot.clone(),
        shape_ok: ok,
        detail: if ok {
            "ok".to_string()
        } else {
            format!("shape {shape:?}, expected [{len}]")
        },
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
    use crate::import::{open_index, read_tensor_f32};

    /// Every landed shard under the workspace weights dir, in shard order.
    /// Missing files are skipped individually so a partial transfer still
    /// tests what is there.
    fn landed_shards() -> Vec<std::path::PathBuf> {
        let dir = std::env::var("QWEN_WEIGHTS_DIR")
            .unwrap_or_else(|_| "/srv/m-sdd/unifur/weights/qwen38-bf16".to_string());
        let mut out = Vec::new();
        for i in 1..=12 {
            let path = std::path::PathBuf::from(format!("{dir}/model-{i:05}-of-00012.safetensors"));
            if path.is_file() {
                out.push(path);
            }
        }
        let mtp = std::path::PathBuf::from(format!("{dir}/model-mtp-restored.safetensors"));
        if mtp.is_file() {
            out.push(mtp);
        }
        out
    }

    fn dims() -> QwenArchDims {
        QwenArchDims::qwen38()
    }

    /// First landed shard, if present (env `QWEN_SHARD1` else the workspace
    /// default). Skips with a note while the 55 GB transfer is in flight.
    fn real_shard1() -> Option<std::path::PathBuf> {
        let candidates = [
            std::env::var("QWEN_SHARD1").unwrap_or_default(),
            "/srv/m-sdd/unifur/weights/qwen38-bf16/model-00001-of-00012.safetensors".to_string(),
        ];
        candidates
            .iter()
            .find(|p| !p.is_empty() && std::path::Path::new(p.as_str()).is_file())
            .map(std::path::PathBuf::from)
    }

    #[test]
    fn test_all_landed_shards_audit_clean() {
        let shards = landed_shards();
        if shards.is_empty() {
            eprintln!("skip: no weight shards landed yet");
            return;
        }
        let d = dims();
        let mut tensors_total = 0usize;
        let mut decoded = 0usize;
        for path in &shards {
            let index =
                open_index(path).unwrap_or_else(|e| panic!("index {}: {e}", path.display()));
            let entries: Vec<(String, Vec<usize>)> = index
                .tensors
                .iter()
                .map(|t| (t.name.clone(), t.shape.clone()))
                .collect();
            tensors_total += entries.len();
            let reports = audit_tensors(
                &entries,
                &d,
                "model.language_model",
                &["mtp.", "model.visual."],
            )
            .unwrap_or_else(|e| panic!("audit {}: {e}", path.display()));
            let bad: Vec<_> = reports.iter().filter(|r| !r.shape_ok).collect();
            assert!(
                bad.is_empty(),
                "{} shape mismatches in {}: {bad:?}",
                bad.len(),
                path.display()
            );
            // Decode one small tensor per shard (biases/norms, KBs): proves
            // the ranged-decode path on real bytes without multi-GB allocs.
            if let Some(meta) = index
                .tensors
                .iter()
                .find(|t| t.byte_len().unwrap_or(u64::MAX) <= 1 << 20)
            {
                let tensor = read_tensor_f32(path, &index, &meta.name)
                    .unwrap_or_else(|e| panic!("decode {} in {}: {e}", meta.name, path.display()));
                assert!(
                    tensor.data.iter().all(|v| v.is_finite()),
                    "non-finite weights in {}",
                    meta.name
                );
                decoded += 1;
            }
        }
        assert!(tensors_total > 0);
        eprintln!("audited {tensors_total} tensors, decoded {decoded} spot-checks");
    }

    #[test]
    fn test_shard1_header_indexes_and_audits_clean() {
        let Some(path) = real_shard1() else {
            eprintln!("skip: shard 1 not landed yet");
            return;
        };
        let index = open_index(&path).expect("index a real 2.5 GB shard header");
        assert!(!index.is_empty());
        let entries: Vec<(String, Vec<usize>)> = index
            .tensors
            .iter()
            .map(|t| (t.name.clone(), t.shape.clone()))
            .collect();
        let reports = audit_tensors(
            &entries,
            &dims(),
            "model.language_model",
            &["mtp.", "model.visual."],
        )
        .expect("audit must not fail structurally");
        assert_eq!(reports.len(), entries.len());
        let bad: Vec<_> = reports.iter().filter(|r| !r.shape_ok).collect();
        assert!(bad.is_empty(), "shape mismatches: {bad:?}");
        // Shard 1 holds the untied lm_head alone: ranged-decode one row
        // (10 KB of a 2.5 GB file) and require finite, nonzero weights.
        let head = read_tensor_f32(&path, &index, "lm_head.weight").expect("ranged head read");
        assert_eq!(head.shape, vec![248320, 5120]);
        let row: Vec<f32> = head.data[..5120].to_vec();
        assert!(row.iter().all(|v| v.is_finite()), "non-finite head weights");
        assert!(row.iter().any(|v| *v != 0.0), "all-zero head row");
    }

    /// Real shard map (`model.safetensors.index.json`): validates names,
    /// not shapes (shapes live in shard headers, audited per shard by
    /// [`audit_tensors`] once files land). Skips with a note where the file
    /// is absent; set `QWEN_INDEX_JSON` to point at one.
    fn real_index_names() -> Option<Vec<String>> {
        let candidates = [
            std::env::var("QWEN_INDEX_JSON").unwrap_or_default(),
            "/srv/m-sdd/unifur/weights/qwen38-bf16/model.safetensors.index.json".to_string(),
        ];
        let path = candidates
            .iter()
            .find(|p| !p.is_empty() && std::path::Path::new(p.as_str()).exists())?;
        let text = std::fs::read_to_string(path.as_str()).ok()?;
        let value: serde_json::Value = serde_json::from_str(&text).ok()?;
        value
            .get("weight_map")?
            .as_object()
            .map(|map| map.keys().cloned().collect())
    }

    #[test]
    fn test_remap_covers_the_real_shard_map() {
        let Some(names) = real_index_names() else {
            eprintln!("skip: no model.safetensors.index.json (weights still downloading?)");
            return;
        };
        let d = dims();
        let prefix = "model.language_model";
        let mut text = 0usize;
        let mut mtp = 0usize;
        let mut visual = 0usize;
        let mut per_layer: std::collections::HashMap<usize, Vec<QwenPart>> =
            std::collections::HashMap::new();
        for name in &names {
            match parse_name(name, prefix).expect("parse must not fail on real names") {
                QwenSlot::Layer { idx, part } => {
                    text += 1;
                    per_layer.entry(idx).or_default().push(part);
                }
                QwenSlot::Foreign(_) => {
                    // MTP heads and the vision tower are allow-listed skips.
                    if name.contains("mtp.") {
                        mtp += 1;
                    } else if name.contains("model.visual.") {
                        visual += 1;
                    } else {
                        panic!("unexpected foreign tensor {name:?}");
                    }
                }
                _ => text += 1,
            }
        }
        // 851 text tensors (48 linear x 14 + 16 full x 11 + embed/norm/head),
        // 15 MTP, 333 vision: 1199 total. Any drift here is a real repo or
        // table change, not noise -- investigate, don't renumber blindly.
        assert_eq!(names.len(), 1199, "shard map grew or shrank");
        assert_eq!((mtp, visual), (15, 333), "MTP + vision inventory changed");
        assert_eq!(text, 48 * 14 + 16 * 11 + 3, "text-trunk inventory changed");
        for idx in 0..d.num_layers {
            let mut parts = per_layer.remove(&idx).unwrap_or_default();
            parts.sort_by_key(|p| format!("{p:?}"));
            let mut want = if d.is_full_attention(idx) {
                vec![
                    QwenPart::FullQ,
                    QwenPart::FullK,
                    QwenPart::FullV,
                    QwenPart::FullO,
                    QwenPart::FullQNorm,
                    QwenPart::FullKNorm,
                ]
            } else {
                vec![
                    QwenPart::LinearQkv,
                    QwenPart::LinearZ,
                    QwenPart::LinearB,
                    QwenPart::LinearA,
                    QwenPart::LinearConv,
                    QwenPart::LinearGateNorm,
                    QwenPart::LinearOut,
                    QwenPart::DeltaDtBias,
                    QwenPart::DeltaALog,
                ]
            };
            want.extend([
                QwenPart::MlpGate,
                QwenPart::MlpUp,
                QwenPart::MlpDown,
                QwenPart::InputNorm,
                QwenPart::PostNorm,
            ]);
            want.sort_by_key(|p| format!("{p:?}"));
            assert_eq!(parts, want, "layer {idx} part inventory mismatch");
        }
    }

    #[test]
    fn test_qwen38_matches_downloaded_config() {
        // Transcribed from the target repo's config.json text_config
        // (2026-09-29): any drift between this preset and that file must be
        // a conscious edit here, not silent rot.
        let d = dims();
        assert_eq!(
            (d.num_layers, d.hidden_size, d.intermediate_size),
            (64, 5120, 17408)
        );
        assert_eq!((d.num_q_heads, d.num_kv_heads, d.head_dim), (24, 4, 256));
        assert_eq!(d.vocab_size, 248320);
        assert_eq!(
            (d.linear_k_groups, d.linear_v_heads, d.linear_head_dim),
            (16, 48, 128)
        );
        assert_eq!(d.num_v_heads(), 48);
        assert_eq!((d.key_dim(), d.value_dim()), (2048, 6144));
        // Interval convention: every 4th layer (1-based) is full attention.
        assert!(!d.is_full_attention(0) && d.is_full_attention(3) && d.is_full_attention(63));
    }

    #[test]
    fn test_every_name_class_parses() {
        // One of each, with the model prefix this repo's shards are expected
        // to use (confirmed against shard 1 when it lands; both tried).
        for prefix in ["model", "language_model"] {
            let p = |tail: &str| parse_name(&format!("{prefix}.{tail}"), prefix).unwrap();
            assert!(matches!(p("embed_tokens.weight"), QwenSlot::Embed));
            assert!(matches!(p("norm.weight"), QwenSlot::FinalNorm));
            assert!(matches!(
                p("layers.7.linear_attn.in_proj_qkv.weight"),
                QwenSlot::Layer {
                    idx: 7,
                    part: QwenPart::LinearQkv
                }
            ));
            assert!(matches!(
                p("layers.7.linear_attn.conv1d.bias"),
                QwenSlot::Layer {
                    idx: 7,
                    part: QwenPart::LinearConvBias
                }
            ));
            assert!(matches!(
                p("layers.7.linear_attn.A_log"),
                QwenSlot::Layer {
                    idx: 7,
                    part: QwenPart::DeltaALog
                }
            ));
            assert!(matches!(
                p("layers.3.self_attn.q_norm.weight"),
                QwenSlot::Layer {
                    idx: 3,
                    part: QwenPart::FullQNorm
                }
            ));
            assert!(matches!(
                p("layers.0.mlp.down_proj.weight"),
                QwenSlot::Layer {
                    idx: 0,
                    part: QwenPart::MlpDown
                }
            ));
            assert!(matches!(
                p("layers.0.input_layernorm.weight"),
                QwenSlot::Layer {
                    idx: 0,
                    part: QwenPart::InputNorm
                }
            ));
        }
        // vLLM-internal fused names must NOT parse as text slots: they never
        // appear in checkpoints, and accepting them would mis-split layers.
        assert!(matches!(
            parse_name("model.layers.7.linear_attn.in_proj_qkvz.weight", "model").unwrap(),
            QwenSlot::Foreign(_)
        ));
        assert_eq!(
            parse_name("lm_head.weight", "model").unwrap(),
            QwenSlot::LmHead
        );
        // A non-numeric layer index under a matching root is malformed.
        assert!(parse_name("model.layers.x.mlp.weight", "model").is_err());
        // A name outside the root is foreign, not malformed.
        assert!(matches!(
            parse_name("other.layers.0.mlp.weight", "model").unwrap(),
            QwenSlot::Foreign(_)
        ));
    }

    #[test]
    fn test_expected_shapes_match_qwen38_dims() {
        let d = dims();
        let h = 5120;
        // PyTorch [out, in]: in_proj_qkv [2*2048 + 6144 = 10240, 5120].
        assert_eq!(QwenPart::LinearQkv.expected_shape(&d), vec![10240, h]);
        assert_eq!(QwenPart::LinearZ.expected_shape(&d), vec![6144, h]);
        assert_eq!(QwenPart::LinearB.expected_shape(&d), vec![48, h]);
        // Depthwise conv weight [channels, 1, K], channels = 2*2048 + 6144.
        assert_eq!(QwenPart::LinearConv.expected_shape(&d), vec![10240, 1, 4]);
        assert_eq!(QwenPart::LinearGateNorm.expected_shape(&d), vec![128]);
        assert_eq!(QwenPart::LinearOut.expected_shape(&d), vec![h, 6144]);
        assert_eq!(QwenPart::DeltaDtBias.expected_shape(&d), vec![48]);
        assert_eq!(QwenPart::FullQ.expected_shape(&d), vec![2 * 24 * 256, h]);
        assert_eq!(QwenPart::FullK.expected_shape(&d), vec![4 * 256, h]);
        assert_eq!(QwenPart::FullO.expected_shape(&d), vec![h, 24 * 256]);
        assert_eq!(QwenPart::MlpDown.expected_shape(&d), vec![h, 17408]);
    }

    #[test]
    fn test_audit_accepts_full_layer_and_refuses_strays() {
        let d = dims();
        // A complete linear layer 0 plus globals, PyTorch [out, in] shapes.
        let entries = vec![
            ("model.embed_tokens.weight".to_string(), vec![248320, 5120]),
            ("model.norm.weight".to_string(), vec![5120]),
            ("lm_head.weight".to_string(), vec![248320, 5120]),
            (
                "model.layers.0.linear_attn.in_proj_qkv.weight".to_string(),
                vec![10240, 5120],
            ),
            (
                "model.layers.0.linear_attn.in_proj_z.weight".to_string(),
                vec![6144, 5120],
            ),
            (
                "model.layers.0.linear_attn.in_proj_b.weight".to_string(),
                vec![48, 5120],
            ),
            (
                "model.layers.0.linear_attn.in_proj_a.weight".to_string(),
                vec![48, 5120],
            ),
            (
                "model.layers.0.linear_attn.conv1d.weight".to_string(),
                vec![10240, 1, 4],
            ),
            (
                "model.layers.0.linear_attn.norm.weight".to_string(),
                vec![128],
            ),
            (
                "model.layers.0.linear_attn.out_proj.weight".to_string(),
                vec![5120, 6144],
            ),
            ("model.layers.0.linear_attn.dt_bias".to_string(), vec![48]),
            ("model.layers.0.linear_attn.A_log".to_string(), vec![48]),
            (
                "model.layers.0.mlp.gate_proj.weight".to_string(),
                vec![17408, 5120],
            ),
            (
                "model.layers.0.mlp.up_proj.weight".to_string(),
                vec![17408, 5120],
            ),
            (
                "model.layers.0.mlp.down_proj.weight".to_string(),
                vec![5120, 17408],
            ),
            (
                "model.layers.0.input_layernorm.weight".to_string(),
                vec![5120],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight".to_string(),
                vec![5120],
            ),
            (
                "model.layers.3.self_attn.q_proj.weight".to_string(),
                vec![12288, 5120],
            ),
            (
                "model.layers.3.self_attn.q_norm.weight".to_string(),
                vec![256],
            ),
            // Allow-listed foreign (MTP-style): reported, not failed.
            ("mtp.head.weight".to_string(), vec![5120, 5120]),
        ];
        let reports = audit_tensors(&entries, &d, "model", &["mtp"]).unwrap();
        assert!(reports.iter().all(|r| r.shape_ok), "all shapes must pass");
        assert_eq!(reports.len(), entries.len());
        // A linear part on a full layer (3) must fail the audit.
        let bad = vec![(
            "model.layers.3.linear_attn.in_proj_qkv.weight".to_string(),
            vec![5120, 10240],
        )];
        assert!(audit_tensors(&bad, &d, "model", &[]).is_err());
        // A non-allow-listed foreign name must fail, not vanish.
        let stray = vec![("mystery.weight".to_string(), vec![8])];
        assert!(audit_tensors(&stray, &d, "model", &["mtp"]).is_err());
        // A wrong shape is reported (not panicked on).
        let wrong = vec![(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            vec![5120, 17407],
        )];
        let rep = audit_tensors(&wrong, &d, "model", &[]).unwrap();
        assert!(!rep[0].shape_ok);
    }
}
