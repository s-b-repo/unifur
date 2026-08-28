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
//! - [`corpus`]: pre-tokenized text corpora, in memory or streamed
//! - [`lm`]: causal language model over the shared trunk
//! - [`schedule`]: LR schedules, EMA, gradient accumulation and clipping
//! - [`reweight`]: per-sigma uncertainty weighting and importance sampling
//! - [`train`]: block-wise training loop
//! - [`checkpoint`]: content-addressed checkpoint saving
//! - [`solver`]: ODE solvers (Euler / Heun / DDIM / DPM-Solver++ 2M & 3M)
//! - [`multi_block`]: sequential / parallel / hybrid / adaptive strategies
//! - [`precision`]: reduced-precision emulation + mixed-precision policy
//! - [`quality`]: quality gates for sampling and batch filtering
//! - [`accuracy`]: guidance, logit normalization, ensembling, compute scaling
//! - [`quantize`]: NF4 blockwise quantization + LoRA adapters (QLoRA)
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

pub mod accuracy;
pub mod adaptive;
pub mod checkpoint;
pub mod cifar;
pub mod consistency;
pub mod corpus;
pub mod data;
pub mod dblock;
pub mod distill;
pub mod expert_index;
pub mod flow;
pub mod infer;
pub mod lm;
pub mod logging;
pub mod loopgraph;
pub mod moe;
pub mod mosme;
pub mod multi_block;
pub mod planner;
pub mod precision;
pub mod profile;
pub mod quality;
pub mod quantize;
pub mod rawdata;
pub mod reweight;
pub mod schedule;
pub mod sigma;
pub mod solver;
pub mod stats;
pub mod tensor_ext;
pub mod tinyimagenet;
pub mod tokenizer;
pub mod train;
pub mod verify;
pub mod vit;
