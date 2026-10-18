# Research-backed improvements: Unifur 6-stage pipeline

1. Shared vector space (VL-JEPA/CLIP + topological alignment).
2. Language prompt -> typed scene graph (LINC/Logic-LM pattern).
3. Generation: structure branch + diffusion content branch (CommonScenes-style).
4. Training efficiency: CoT distillation + Diffusion of Thought.
5. Optional post-training RL (GDPO).
6. Output translation of internal vector reasoning to plain language.

## Stage 3/4 references

### Diffusion of Thought (DoT) — NeurIPS 2024
- Paper: "Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models"
- arXiv: https://arxiv.org/abs/2402.07754
- Code: https://github.com/HKUNLP/diffusion-of-thoughts
- Techniques: reasoning steps diffuse over time; scheduled sampling; coupled sampling for multi-pass robustness; conditional ODE solver reduces inference to 8-64 steps; self-consistency decoding.

### Chain-of-Thought Distillation
- "Chain-of-Thought Curriculum Distillation: Teaching Smaller Models to Reason Step-by-Step" (ACM 2025 ISAIE): https://dl.acm.org/doi/10.1145/3775073.3775200
- Overview: https://medium.com/@milesk_33/scaling-down-smarting-up-distillation-techniques-that-preserve-reasoning-in-tiny-transformers-7abeb507fa11
- Core idea: teacher generates step-by-step reasoning traces; fine-tune on reasoning path + final answer.

### Group Diffusion Policy Optimization (GDPO) — ICLR 2026
- arXiv: https://arxiv.org/abs/2510.08554
- Core idea: RL for diffusion LMs; variance decomposition of sequence-level ELBO; group sampling optimized by correctness reward.

### GPU-budget references
- "Can A Gamer Train A Mathematical Reasoning Model?": https://arxiv.org/html/2506.08935
- "d1: Scaling Reasoning in Diffusion Large Language Models": https://arxiv.org/pdf/2504.12216

### Related neuro-symbolic repos (candidate reading, not dependencies)
- neurosymbolic-diffusion (HEmile / april-tools, NeurIPS 2025): discrete diffusion + neurosymbolic predictors; closest to this crate's diffusion architecture.
- LINC (benlipkin, EMNLP 2023): LM as translator into first-order logic, then a theorem prover; 15B open model beat GPT-4+CoT by 10% on one benchmark.
- Logic-LM (teacherpeterpan): NL -> symbolic form -> solver.
- neusire: neuro-symbolic visual reasoning via scene-graph enrichment.
- CommonScenes (NeurIPS 2023): diffusion model generating scenes directly from graph structure; closest match to the two-branch generation stage. Trains on a single 3090 at reduced batch size.

