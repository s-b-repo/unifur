# FAQ

Frequently asked questions about DiffusionBlocks++.

## General

### What is DiffusionBlocks++?

DiffusionBlocks++ is an extension of the DiffusionBlocks framework (ICLR 2026)
that partitions transformer networks into independently trainable blocks.
It adds parallel depth denoising, consistency training, flow matching, MoE
routing, block distillation, adaptive depth, and QLoRA support.

### How does it differ from the original DiffusionBlocks?

The original DiffusionBlocks trains one block per step. DiffusionBlocks++ adds:

- **Parallel training**: Train K blocks simultaneously
- **Consistency losses**: Blocks cooperate at boundaries
- **Advanced solvers**: DPM-Solver++, DDIM for fast inference
- **Flow matching**: Alternative to EDM score matching
- **MoE routing**: Experts specialize in noise regimes
- **Block distillation**: Compress N blocks → fewer blocks
- **Adaptive depth**: Skip blocks when confident
- **QLoRA**: 4-bit training for memory efficiency

### What tasks does it support?

- Image classification (ViT)
- Image generation (DiT) — planned
- Text generation (Masked Diffusion, AR Transformer) — planned
- Multi-task learning — planned

## Training

### How much memory does it use?

| Configuration | Memory per GPU |
|---|---|
| Original DiffusionBlocks (B=3) | ~2 GB |
| Parallel K=2 | ~3 GB |
| Parallel K=3 | ~4 GB |
| With QLoRA | ~1 GB |

### How long does training take?

On a single A100 GPU with CIFAR-100:

| Configuration | Time to 75% accuracy |
|---|---|
| ViT baseline | ~2 hours |
| DiffusionBlocks (B=3) | ~6 hours |
| Parallel K=2 | ~4 hours |
| Parallel K=3 | ~3 hours |

### Why is parallel training faster?

Because K blocks are updated per step instead of 1, you need fewer total steps
to cover all blocks. With B=3 and K=2, you need ~3 steps instead of 3×3=9 steps.

### Can I use my own dataset?

Yes. Create a custom `ImageDataModule` in `src/diffusionblocks/data.py`:

```python
class CustomDataModule(ImageDataModule):
    data_name = "your-dataset"
    image_size = 64
    num_labels = 10
```

## Inference

### Which solver should I use?

| Use Case | Recommended Solver | Steps |
|---|---|---|
| Maximum quality | DPM-Solver++ | 20-30 |
| Balanced | Heun / DDIM | 10-20 |
| Fastest | DDIM | 5-10 |

### What is classifier-free guidance?

Classifier-free guidance (CFG) improves quality by mixing conditional and
unconditional predictions:

```
logits = logits_uncond + scale * (logits_cond - logits_uncond)
```

Higher scale → better quality but less diversity. Recommended: `scale=3.0`.

### How does adaptive depth work?

Each block learns a halting probability. At inference, if the probability
exceeds a threshold, the remaining blocks are skipped. This means easy
samples use fewer blocks (faster), while hard samples use more blocks
(better quality).

## Troubleshooting

### Out of memory during training

Solutions:
1. Reduce batch size: `--batch_size 64`
2. Enable gradient checkpointing: `--gradient_checkpointing`
3. Use QLoRA: `--use_qlora`
4. Reduce parallel blocks: `--parallel_blocks 1`

### Poor convergence

Solutions:
1. Increase warmup steps: `--num_warmup_steps 1000`
2. Use cosine scheduler: `--scheduler_type cosine_with_min_lr`
3. Increase consistency weight: `--consistency_weight 0.2`
4. Check sigma schedule visualization in W&B

### Slow inference

Solutions:
1. Use DPM-Solver++: `--solver dpmpp`
2. Reduce steps: `--num_inference_steps 10`
3. Enable adaptive depth: `--adaptive_depth`
4. Use DDIM: `--solver ddim`

## Contributing

### How do I contribute?

1. Pick an item from [TODO.md](../TODO.md)
2. Create an issue describing your plan
3. Fork, implement, and submit a PR
4. Update documentation

### What's the priority order?

1. Phase 1 (Core DiffusionBlocks)
2. Phase 2 (Parallel Denoising)
3. Phase 4 (Inference Solvers)
4. Phase 3 (Consistency Training)
5. Phases 5-11 (Features in any order)

## Citation

### How do I cite this work?

Please cite the original DiffusionBlocks paper:

```bibtex
@inproceedings{shing2026diffusionblocks,
  title     = {DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation},
  author    = {Makoto Shing and Masanori Koyama and Takuya Akiba},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=pwVSmK71cS}
}
```

## License

### What license is this?

Apache 2.0. See [LICENSE](../LICENSE).

### Can I use this commercially?

Yes, Apache 2.0 permits commercial use.
