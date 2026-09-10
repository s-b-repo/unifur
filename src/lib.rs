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
//! # Layout
//!
//! - [`stats`]: scalar statistics helpers (erf, normal CDF / quantile)
//! - [`sigma`]: EDM / dblock noise schedules and preconditioning
//! - [`vit`]: ViT-DiT backbone with adaLN-zero timestep conditioning
//! - [`dblock`]: block-wise denoising classifier (train + Euler sampling)
//! - [`data`]: batch abstraction, synthetic dataset
//! - [`rawdata`]: fixed-record image datasets + streaming/coalesced I/O
//! - [`cifar`]: CIFAR-100 binary-format dataset
//! - [`tinyimagenet`]: Tiny ImageNet raw-format dataset
//! - [`tokenizer`]: dependency-free byte-level tokenizer
//! - [`antipattern`]: anti-pattern rules and per-token labels for negative supervision
//! - [`corpus`]: pre-tokenized text corpora, in memory or streamed
//! - [`lm`]: causal language model over the shared trunk
//! - [`hybrid`]: per-layer attention modes (dense, sliding, retrieval,
//!   linear, learned), rotary positions and per-mode decode state
//! - [`routing`]: the per-token routing state carried through the layers,
//!   router kinds and routing-locality diagnostics
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
//! - [`quality_coder`]: scaffolding for the agentic code refiner
//!   described in `docs/Quality-Coder.md`. Data adapters + eval harness
//!   only; no model is loaded or trained yet.
//! - [`consistency`]: boundary / self / trajectory consistency losses
//! - [`distill`]: teacher/student block distillation (KL + trajectory)
//! - [`flow`]: rectified-flow objective and sampler
//! - [`moe`]: top-k mixture-of-experts layer with load balancing
//! - [`expert_index`]: the expert manifest an inference engine routes from
//! - [`mosme`]: boxes of specialized micro experts, two-level routing
//! - [`adaptive`]: halting head + early-exit logic
//! - [`loopgraph`]: dynamic loop-graph execution (skip / loop / budget)
//! - [`profile`]: scope-level timing harness
//! - [`planner`]: next-step and path prediction (beam search over trajectories)
//! - [`infer`]: batched offline inference API
//! - [`logging`]: JSONL metrics logger
//! - [`verify`]: numerical certificate suite (the quality gate)

pub mod ablation;
pub mod accuracy;
pub mod adaptive;
pub mod audit;
pub mod antipattern;
pub mod checkpoint;
pub mod cifar;
pub mod codequality;
pub mod consistency;
pub mod corpus;
pub mod cost;
pub mod data;
pub mod dblock;
pub mod distill;
pub mod experiment;
pub mod expert_index;
pub mod flow;
pub mod heretic;
pub mod hybrid;
pub mod infer;
pub mod lm;
pub mod merge;
pub mod mix;
pub mod logging;
pub mod loopgraph;
pub mod moe;
pub mod mosme;
pub mod multi_block;
pub mod planner;
pub mod policy;
pub mod precision;
pub mod profile;
pub mod quality;
pub mod quality_coder;
pub mod quantize;
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
