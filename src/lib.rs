//! DiffusionBlocks++ in Rust.
//!
//! Block-wise neural network training via diffusion interpretation, ported
//! from the reference PyTorch implementation (SakanaAI/DiffusionBlocks) to
//! the Burn deep-learning framework.
//!
//! # Layout
//!
//! - [`stats`]: scalar statistics helpers (erf, normal CDF / quantile)
//! - [`sigma`]: EDM / dblock noise schedules and preconditioning
//! - [`vit`]: ViT-DiT backbone with adaLN-zero timestep conditioning
//! - [`dblock`]: block-wise denoising classifier (train + Euler sampling)
//! - [`data`]: batch abstraction, synthetic dataset
//! - [`cifar`]: CIFAR-100 binary-format dataset
//! - [`train`]: block-wise training loop
//! - [`checkpoint`]: content-addressed checkpoint saving
//! - [`solver`]: ODE solvers (Euler / Heun / DDIM / DPM-Solver++2M)
//! - [`multi_block`]: sequential / parallel / hybrid / adaptive strategies
//! - [`quality`]: quality gates for sampling and batch filtering
//! - [`consistency`]: boundary / self / trajectory consistency losses
//! - [`flow`]: rectified-flow objective and sampler
//! - [`moe`]: top-k mixture-of-experts layer with load balancing
//! - [`adaptive`]: halting head + early-exit logic
//! - [`logging`]: JSONL metrics logger

pub mod adaptive;
pub mod checkpoint;
pub mod cifar;
pub mod consistency;
pub mod data;
pub mod dblock;
pub mod flow;
pub mod logging;
pub mod moe;
pub mod multi_block;
pub mod quality;
pub mod sigma;
pub mod solver;
pub mod stats;
pub mod tensor_ext;
pub mod train;
pub mod vit;
