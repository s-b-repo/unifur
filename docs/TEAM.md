# Team

## Maintainers

- DiffusionBlocks++ Contributors

## Acknowledgements

- Original DiffusionBlocks authors: Makoto Shing, Masanori Koyama, Takuya Akiba (Sakana AI)
- HuggingFace Transformers team (the ViT the backbone is ported from)
- Karras et al. / NVIDIA (EDM preconditioning and schedules)
- Lu et al. (DPM-Solver++), Song et al. (DDIM)
- The [Burn](https://burn.dev) team (the tensor framework this crate is built on)

## Contributing

See [`TODO.md`](../TODO.md) for open items and their blockers.

Before opening a PR:

```bash
cargo test --all && cargo clippy --all-targets && dblocks verify
```

New mathematical claims belong in `src/verify.rs` as certificates, not as
ad-hoc tests — see [Quality Gate](Quality-Gate.md).
