//! Evaluate a trained CIFAR-100 MoSME diffusion classifier, or classify one image.

use anyhow::{ensure, Context, Result};
use burn::tensor::{backend::Backend, Tensor};
use burn_ndarray::NdArray;
use clap::Parser;
use diffusionblocks::{
    checkpoint, cifar,
    data::{CIFAR100_MEAN, CIFAR100_STD},
    dblock::{DblockClassifier, DblockConfig},
    expert_index::MosmeSpec,
    infer::{InferenceConfig, InferenceEngine},
    vit::{MosmeTrunkConfig, ViTDiTConfig},
};
use rand::{rngs::StdRng, SeedableRng};
use serde_json::{json, Value};
use std::path::PathBuf;

type Cpu = NdArray<f32>;
const PLANE: usize = 32 * 32;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    bundle: PathBuf,
    /// A 32x32 RGB image stored as three unsigned byte planes, red first.
    #[arg(long, conflicts_with = "test")]
    image: Option<PathBuf>,
    /// Score the packaged, class-balanced subset of the official test split.
    #[arg(long)]
    test: bool,
}

fn normalized(pixels: &[u8]) -> impl Iterator<Item = f32> + '_ {
    pixels.iter().enumerate().map(|(index, &value)| {
        let channel = index / PLANE;
        (f32::from(value) / 255.0 - CIFAR100_MEAN[channel]) / CIFAR100_STD[channel]
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let card: Value =
        serde_json::from_str(&std::fs::read_to_string(args.bundle.join("model.json"))?)?;
    let weights = card
        .get("checkpoint")
        .and_then(Value::as_str)
        .context("missing checkpoint name")?;
    let names = std::fs::read_to_string(args.bundle.join("fine_label_names.txt"))?;
    let names: Vec<&str> = names.lines().collect();
    ensure!(names.len() == 100, "expected 100 CIFAR-100 class names");
    let spec = MosmeSpec::read(&args.bundle.join("experts.json"))?;
    let trunk =
        ViTDiTConfig::cifar(100).with_mosme(MosmeTrunkConfig::new(spec).with_every_n_layers(2));
    let config = DblockConfig {
        num_blocks: 3,
        ..DblockConfig::default()
    };
    let device = Default::default();
    Cpu::seed(&device, 42);
    let model = DblockClassifier::<Cpu>::new(&trunk, &config, &device)?;
    let model = checkpoint::load(model, &args.bundle.join(weights), &device)?;
    let engine = InferenceEngine::new(
        model,
        InferenceConfig {
            batch_size: 16,
            ..InferenceConfig::default()
        },
    );
    let mut rng = StdRng::seed_from_u64(42);

    if args.test {
        let split = cifar::read_split(&args.bundle.join("heldout.bin"))?;
        ensure!(!split.is_empty(), "the held-out image split is empty");
        let pixels: Vec<f32> = (0..split.len())
            .flat_map(|index| normalized(split.record(index)))
            .collect();
        let inputs = Tensor::<Cpu, 1>::from_floats(pixels.as_slice(), &device).reshape([
            split.len(),
            3,
            32,
            32,
        ]);
        let output = engine.classify(inputs, &mut rng);
        let correct = output
            .labels
            .iter()
            .zip(&split.labels)
            .filter(|(predicted, actual)| usize::try_from(**actual).ok() == Some(**predicted))
            .count();
        println!(
            "{}",
            json!({"cases": split.len(), "correct": correct,
            "accuracy": correct as f64 / split.len() as f64, "predictions": output.labels})
        );
    } else {
        let image = args.image.unwrap_or_else(|| args.bundle.join("demo.rgb"));
        let bytes = std::fs::read(&image).with_context(|| format!("read {}", image.display()))?;
        ensure!(
            bytes.len() == 3 * PLANE,
            "expected a 3072-byte planar RGB image"
        );
        let pixels: Vec<f32> = normalized(&bytes).collect();
        let input =
            Tensor::<Cpu, 1>::from_floats(pixels.as_slice(), &device).reshape([1, 3, 32, 32]);
        let output = engine.classify(input, &mut rng);
        let label = output
            .labels
            .first()
            .copied()
            .context("empty classifier output")?;
        let name = names
            .get(label)
            .context("classifier returned a class outside CIFAR-100")?;
        println!(
            "{}",
            json!({"label": label, "class": name, "confidence": output.confidence.first()})
        );
    }
    Ok(())
}
