# FAQ

Frequently asked questions about DiffusionBlocks++.

> **In this repository.** Answers below that reference Python APIs describe
> the original design spec. The implementation is a Rust crate — see
> [Home](Home.md) for the module map and [`TODO.md`](../TODO.md) for what is
> and is not implemented.


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

- Image classification (ViT) — implemented
- **Causal language modeling** — implemented (`lm.rs`), over the same trunk:
  byte tokenizer, tied head, KV cache, streamed corpora. See
  [Language Modeling](Language-Modeling.md). Loading pretrained Llama- or
  Mistral-class weights is out of scope; the trunk trains from scratch
- Image generation (DiT) — planned
- Masked-diffusion text generation — planned
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

Guidance improves quality by extrapolating away from the unconditional
prediction:

```
x0 = x0_uncond + scale * (x0_cond - x0_uncond)
```

Higher scale → sharper conditioning, less diversity, and past some point worse
fidelity. `dblocks sample --guidance 3.0` is a reasonable starting point;
`--guidance-rescale 0.7` contains the norm inflation a strong scale causes.

Two things worth knowing here. "Unconditional" means a **zero image**, not a
learned null embedding — this model has no null token, and a learned one is the
natural upgrade if guidance proves worth training for. And `--guidance 1.0`
returns the conditional estimate *bitwise*, not via `u + 1*(c - u)`, which is
equal only in exact arithmetic. Details:
[Accuracy Improvements](Accuracy-Improvements.md).

### Can it plan ahead instead of taking the greedy step?

Yes, on both paths. `dblocks sample --planned` scores candidate
`(sigma, span)` pairs and short rollouts of what follows them;
`dblocks lm generate --lookahead N` scores candidate token *continuations* and
commits only the first token.

Both are certified to reduce **exactly** to the greedy policy at depth 0, and
both enforce a hard evaluation budget — the allowance is handed to the scoring
function before it does the work, not applied afterwards. See
[Next-Step Planning](Next-Step-Planning.md).

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

### Why is one block's loss two orders of magnitude larger than another's?

Because the EDM weight `w(sigma) = (sigma^2 + sigma_d^2)/(sigma*sigma_d)^2`
diverges as sigma goes to zero, and the block owning the low-noise end inherits
that. This repository measured 13.4 for block 0 against 1909.8 for block 2 — a
~140x spread — and it is expected from the objective, not a bug.

It still matters: a shared trunk trained on a sum of terms that differ by two
orders of magnitude is, in effect, trained almost entirely on the largest one.
Three flags address it:

```bash
dblocks train --normalize-block-loss     # equalize on the geometric mean
dblocks train --uncertainty 1.0          # learn the per-sigma scale and divide it out
dblocks train --importance-bins 16       # spend samples where the loss varies
```

`--uncertainty` is the principled one: at its optimum the gradient becomes that
of *log* loss, which is invariant to any per-sigma rescaling — so no weighting
convention can reintroduce the imbalance. See
[Loss Reduction](Loss-Reduction.md).

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
