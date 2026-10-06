//! DiffusionBlocks++ in Rust.
//!
//! Block-wise neural network training via diffusion interpretation, ported
//! from the reference PyTorch implementation (SakanaAI/DiffusionBlocks) to
//! the Burn deep-learning framework.
//!
//! # Quality gate
//!
//! Every load-bearing mathematical identity in this crate is stated as a
//! theorem and checked as a numerical residual by [`verify`], which
//! `dblocks verify` and the test suite both run. A regression in a solver
//! coefficient, a schedule convention or a preconditioning identity therefore
//! surfaces as a named certificate failure rather than as quietly worse
//! results.
//!
//! # The error-handling contract
//!
//! Two rules, both machine-enforced rather than conventional (see the `[lints]`
//! table in `Cargo.toml`):
//!
//! 1. **No panicking escape hatch in production code.** `unwrap`, `expect`,
//!    `panic!`, `todo!`, `unimplemented!` and `unreachable!` are all denied
//!    outside `#[cfg(test)]`. A fallible path returns
//!    [`anyhow::Result`] and names the value that made it fail.
//! 2. **Tests are exempt, not exempt-by-default.** A `#[cfg(test)]` module may
//!    `unwrap`; that is the point of a test. An exemption is scoped to the
//!    module or file that asked for it, so it cannot leak into the next one.
//!
//! `assert!` is *not* denied, and that is deliberate. An assertion states an
//! invariant the caller established -- "the block index is in range", "these
//! two shapes match" -- and there is no error channel to return through at that
//! point. What is denied is reaching for a runtime value that might be absent.
//!
//! # Layout
//!
//! # Testing exemptions
//!
//! A `#[cfg(test)]` module gets the unwrap family back, and only that module
//! does. The grant is written per module rather than once at the crate root
//! precisely so that adding a production module cannot inherit it.
//!
//! Two things read these grants and would otherwise have to be kept in sync by
//! hand:
//!
//! - `audit-bad-patterns.sh` section C, which distinguishes a grant inside a test
//!   module from one in production code by where it sits in the file.
//! - `crate::cheat`, whose detector does the same check in Rust and reports a
//!   grant in production code as gate tampering.
//!
//! A new test module therefore needs its grant written the same way, or both
//! will report it.
//!
//! - [`stats`]: scalar statistics helpers (erf, normal CDF / quantile)
//! - [`sigma`]: EDM / dblock noise schedules and preconditioning
//! - [`vit`]: ViT-DiT backbone with adaLN-zero timestep conditioning
//! - [`dblock`]: block-wise denoising classifier (train + Euler sampling)
//! - [`deltanet`]: Gated DeltaNet reference core (Phase C of
//!   `docs/Frontier-27B-Plan.md`): the gated delta rule, L2-normalized
//!   queries/keys, causal short convolution, per-pair recurrent state
//! - [`data`]: batch abstraction, synthetic dataset
//! - [`rawdata`]: fixed-record image datasets + streaming/coalesced I/O
//! - [`cifar`]: CIFAR-100 binary-format dataset
//! - [`tinyimagenet`]: Tiny ImageNet raw-format dataset
//! - [`tokenizer`]: dependency-free byte-level tokenizer
//! - [`bpe`]: BPE merge engine over `u32` ids, compatible with HuggingFace
//!   `tokenizer.json` (Phase A of `docs/Frontier-27B-Plan.md`)
//! - [`antipattern`]: anti-pattern rules and per-token labels for negative supervision
//! - [`corpus`]: pre-tokenized text corpora, in memory or streamed
//! - [`lm`]: causal language model over the shared trunk
//! - [`hybrid`]: per-layer attention modes (dense, sliding, retrieval,
//!   linear, learned), rotary positions and per-mode decode state
//! - [`import`]: safetensors reader (header index + ranged `f32` decode)
//!   for Phase B of `docs/Frontier-27B-Plan.md`
//! - [`geometry`]: the geometric reasoner (roadmap Phase 33): a dual-stream
//!   symbolic + geometric model answering by attractor relaxation under a
//!   learned Riemannian metric, trained with the block-diffusion machinery
//! - [`routing`]: the per-token routing state carried through the layers,
//!   router kinds and routing-locality diagnostics
//! - [`blockexec`]: how a batch of independent blocks is executed -- one at a
//!   time, one thread per item, or on a persistent pool -- with the guarantee
//!   that the mode is a performance decision and never a different program
//! - [`cost`]: active parameters, FLOPs and resident state per token, counted
//!   from the configuration
//! - [`schedule`]: LR schedules, EMA, gradient accumulation and clipping
//! - [`reweight`]: per-sigma uncertainty weighting and importance sampling
//! - [`train`]: block-wise training loop
//! - [`checkpoint`]: content-addressed checkpoints and training-state sidecars
//! - [`experiment`]: experiment records with environment, seeds, raw trials and t-intervals
//! - [`sweep`]: hyperparameter grids trained per seed into experiment records
//! - [`mix`]: several datasets or corpora in one run, as a mixture or composite
//! - [`merge`]: weighted averaging of same-architecture checkpoints
//! - [`policy`]: blockers, refusals and signed approvals gating a model's capabilities
//! - [`audit`]: error-propagation audit of a block-wise model
//! - [`solver`]: ODE solvers (Euler / Heun / DDIM / DPM-Solver++ 2M & 3M)
//! - [`multi_block`]: sequential / parallel / hybrid / adaptive strategies
//! - [`precision`]: reduced-precision emulation + mixed-precision policy
//! - [`quality`]: quality gates for sampling and batch filtering
//! - [`codequality`]: per-language code-quality signals, pre-training window
//!   filter, and a regularizer that pulls the loss toward a target score
//! - [`accuracy`]: guidance, logit normalization, ensembling, compute scaling
//! - [`quantize`]: NF4 blockwise quantization + LoRA adapters (QLoRA)
//! - [`qwen`]: Qwen3.8 weight inventory: architecture dims, checkpoint
//!   tensor-name remap and per-shard audit (Phase B/C of
//!   `docs/Frontier-27B-Plan.md`)
//! - [`quality_coder`]: scaffolding for the agentic code refiner
//!   described in `docs/Quality-Coder.md`. Data adapters + eval harness
//!   only; no model is loaded or trained yet.
//! - [`consistency`]: boundary / self / trajectory consistency losses
//! - [`distill`]: teacher/student block distillation (KL + trajectory)
//! - [`flow`]: rectified-flow objective and sampler
//! - [`moe`]: top-k mixture-of-experts layer with load balancing
//! - [`expert_index`]: the expert manifest an inference engine routes from
//! - [`mosme`]: boxes of specialized micro experts, two-level routing
//! - [`peregrine`]: the parallel I/O engine -- mirror-striped reads split
//!   across every drive, batched through io_uring where the kernel allows it,
//!   checked byte-for-byte against a single-drive read
//! - [`adaptive`]: halting head + early-exit logic
//! - [`loopgraph`]: dynamic loop-graph execution (skip / loop / budget)
//! - [`profile`]: scope-level timing harness
//! - [`planner`]: next-step and path prediction (beam search over trajectories)
//! - [`geometry`]: geometric reasoning over a learned Riemannian scene
//!   geometry -- a dual-stream reasoner whose answer is a certified attractor
//!   relaxation, trained with the block-diffusion objectives. See
//!   `docs/Geometric-Reasoning.md`
//! - [`geomkernel`]: the exact rational geometry kernel: scene-graph
//!   saturation to a fixed point and randomized counterexample search. The
//!   model proposes; this proves
//! - [`infer`]: batched offline inference API
//! - [`logging`]: JSONL metrics logger
//! - [`verify`]: numerical certificate suite (the quality gate)
//!
//! # The error-handling contract
//!
//! Two rules, both machine-enforced rather than conventional (see the `[lints]`
//! table in `Cargo.toml`):
//!
//! 1. **No panicking escape hatch in production code.** `unwrap`, `expect`,
//!    `panic!`, `todo!`, `unimplemented!` and `unreachable!` are all denied
//!    outside `#[cfg(test)]`. A fallible path returns
//!    [`anyhow::Result`] and names the value that made it fail.
//! 2. **Tests are exempt, not exempt-by-default.** A `#[cfg(test)]` module may
//!    `unwrap`; that is the point of a test. An exemption is scoped to the
//!    module or file that asked for it, so it cannot leak into the next one.
//!
//! `assert!` is *not* denied, and that is deliberate. An assertion states an
//! invariant the caller established -- "the block index is in range", "these
//! two shapes match" -- and there is no error channel to return through at that
//! point. What is denied is reaching for a runtime value that might be absent.

pub mod ablation;
pub mod accuracy;
pub mod adaptive;
pub mod antipattern;
pub mod audit;
pub mod blockexec;
pub mod bpe;
pub mod cheat;
pub mod checkpoint;
pub mod cifar;
pub mod codegen_eval;
pub mod codequality;
pub mod consistency;
pub mod corpus;
pub mod cost;
pub mod data;
pub mod dblock;
pub mod deltanet;
pub mod distill;
pub mod experiment;
pub mod expert_index;
pub mod flow;

pub mod geom3d;
pub mod geomaugment;
pub mod geombaseline;
pub mod geometry;
pub mod geomkernel;
pub mod geomvision;

pub mod grade;
pub mod heretic;
pub mod hybrid;
pub mod import;
pub mod infer;
pub mod lm;
pub mod logging;
pub mod loopgraph;
pub mod merge;
pub mod mix;
pub mod moe;
pub mod mosme;
pub mod multi_block;
pub mod peregrine;
pub mod planner;
pub mod policy;
pub mod precision;
pub mod profile;
pub mod quality;
pub mod quality_coder;
pub mod quantize;
pub mod qwen;
pub mod qwennet;
pub mod rawdata;
pub mod reweight;
pub mod routing;
pub mod schedule;
pub mod sigma;
pub mod solver;
pub mod stats;
pub mod sweep;
pub mod tensor_ext;
pub mod tinyimagenet;
pub mod tokenizer;
pub mod train;
pub mod verify;
pub mod vit;
