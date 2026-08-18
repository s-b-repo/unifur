# Inference Guide

Inference and evaluation for DiffusionBlocks++ models.

## Basic Inference

```bash
# Evaluate a trained model
uv run python -m diffusionblocks.main test cifar100 \
    --model_type dblock --ckpt_path logs/<run>/last.ckpt
```

## Solver Selection

### Euler (Default)

```bash
uv run python -m diffusionblocks.main test cifar100 \
    --model_type dblock --ckpt_path logs/<run>/last.ckpt \
    --solver euler --num_inference_steps 50
```

### DPM-Solver++ (Recommended)

```bash
uv run python -m diffusionblocks.main test cifar100 \
    --model_type dblock --ckpt_path logs/<run>/last.ckpt \
    --solver dpmpp --num_inference_steps 20
```

### DDIM (Fast)

```bash
uv run python -m diffusionblocks.main test cifar100 \
    --model_type dblock --ckpt_path logs/<run>/last.ckpt \
    --solver ddim --num_inference_steps 10
```

## Classifier-Free Guidance

```bash
# Enable CFG for improved quality
uv run python -m diffusionblocks.main test cifar100 \
    --model_type dblock --ckpt_path logs/<run>/last.ckpt \
    --cfg_scale 3.0 --class_dropout_prob 0.1
```

## Adaptive Depth

```bash
# Enable block skipping
uv run python -m diffusionblocks.main test cifar100 \
    --model_type dblock --ckpt_path logs/<run>/last.ckpt \
    --adaptive_depth --halting_threshold 0.9
```

## Benchmarking

### Solver Comparison

```bash
# Run all solvers and compare
python -m diffusionblocks.benchmark \
    --ckpt_path logs/<run>/last.ckpt \
    --solvers euler,heun,ddim,dpmpp \
    --steps 5,10,20,50
```

### Throughput

```bash
# Measure throughput
python -m diffusionblocks.benchmark \
    --ckpt_path logs/<run>/last.ckpt \
    --batch_size 128 --num_runs 100
```

## Inference API

### Python API

```python
from diffusionblocks import load_model
from diffusionblocks.solvers import get_solver

# Load model
model = load_model(args)
model.load_state_dict(torch.load("logs/<run>/last.ckpt"))
model.eval()

# Create solver
solver = get_solver("dpmpp", num_steps=20)

# Run inference
with torch.no_grad():
    z = torch.randn(batch_size, hidden_size) * sigma_max
    z_final = solver.solve(z_init=z, denoise_fn=model.denoise, pixel_values=pixel_values)
    logits = model.forward_output_embeddings(z_final)
    predictions = logits.argmax(dim=-1)
```

### Serving

```bash
# Start inference server
uv run python -m diffusionblocks.serve \
    --ckpt_path logs/<run>/last.ckpt \
    --port 8000
```

```python
# Client
import requests

response = requests.post("http://localhost:8000/predict", json={
    "image": base64_encoded_image,
    "solver": "dpmpp",
    "num_steps": 20,
})
predictions = response.json()["predictions"]
```
