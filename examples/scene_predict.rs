//! Load a trained spatial model and score unlabeled prompts in its training format.

use anyhow::{Context, Result, ensure};
use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};
use clap::{Parser, ValueEnum};
use diffusionblocks::{
    checkpoint,
    geombaseline::{GeomBaseline, GeomBaselineConfig},
    geometry::{GeomConfig, GeomMeta},
    student::{GeometricStudent, StudentConfig},
    tokenizer::Special,
};
use std::path::{Path, PathBuf};

type Cpu = NdArray<f32>;
type Device = <Cpu as burn::tensor::backend::BackendTypes>::Device;

#[derive(Clone, Copy, ValueEnum)]
enum Kind {
    Baseline,
    Student,
}

#[derive(Parser)]
struct Args {
    #[arg(value_enum)]
    kind: Kind,
    #[arg(long)]
    checkpoint: PathBuf,
    /// GeomMeta for baseline, StudentConfig for student.
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    scene: String,
}

fn read_config<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let source =
        std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&source).with_context(|| format!("parse {}", path.display()))
}

fn tokens(scene: &str, add_bos: bool, device: &Device) -> Tensor<Cpu, 2, Int> {
    let mut ids = Vec::with_capacity(scene.len() + usize::from(add_bos));
    if add_bos {
        ids.push(i64::from(Special::Bos.id()));
    }
    ids.extend(scene.bytes().map(i64::from));
    Tensor::<Cpu, 1, Int>::from_ints(ids.as_slice(), device).reshape([1, ids.len()])
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.scene.len() == 29 && args.scene.is_ascii() && args.scene.ends_with('?'),
        "expected a 29-byte four-point scene, ending in '?'"
    );
    let device = Device::default();
    let logits = match args.kind {
        Kind::Baseline => {
            let meta: GeomMeta = read_config(&args.config)?;
            let geom = GeomConfig {
                scene_len: meta.scene_len,
                answer_tokens: meta.answer_tokens,
                slots: 4,
                ..GeomConfig::default()
            };
            let model = GeomBaseline::<Cpu>::new(&GeomBaselineConfig::matched_to(&geom), &device)?;
            let model = checkpoint::load(model, &args.checkpoint, &device)?;
            model.forward(tokens(&args.scene, false, &device))
        }
        Kind::Student => {
            let config: StudentConfig = read_config(&args.config)?;
            let model = GeometricStudent::<Cpu>::new(&config, &device)?;
            let model = checkpoint::load(model, &args.checkpoint, &device)?;
            let state = model.forward(tokens(&args.scene, true, &device))?;
            let [batch, length, vocab] = state.logits.dims();
            state
                .logits
                .narrow(1, length - 1, 1)
                .reshape([batch, vocab])
        }
    };
    let winner: i64 = logits.argmax(1).into_scalar();
    println!("answer token: {winner}");
    Ok(())
}
