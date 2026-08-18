# Block Distillation

Compress N DiffusionBlocks → fewer blocks for efficient inference using
teacher-student knowledge distillation.

## Overview

After training a large DiffusionBlocks model with many blocks, you may want
to deploy a smaller model with fewer blocks for faster inference.

**Block distillation** transfers knowledge from a teacher (many blocks) to
a student (fewer blocks) by minimizing the difference between their denoising
trajectories.

## Methods

### 1. Trajectory Distillation (MSE)

Match the denoised outputs of teacher and student at every sigma:

```
L_traj = E[ || teacher_denoise(z, σ) - student_denoise(z, σ) ||² ]
```

### 2. Distribution Distillation (KL)

Match the output probability distributions:

```
L_kl = E[ KL(teacher_logits(z, σ) || student_logits(z, σ)) ]
```

### 3. Feature Distillation

Match intermediate hidden states:

```
L_feat = E[ || teacher_hidden(z, σ) - student_hidden(z, σ) ||² ]
```

## Block Mapping

When distilling N → M blocks (N > M), we need to map student blocks to
teacher blocks:

```
Teacher: Block0 → Block1 → Block2 → Block3 → Block4 → Block5
          ↓         ↓         ↓         ↓         ↓         ↓
Student: Block0 --------→ Block1 --------→ Block2
```

Each student block is trained to mimic multiple teacher blocks.

## Training Loop

```python
for batch in dataloader:
    # Forward through both models
    with torch.no_grad():
        teacher_output = teacher(pixel_values, z, sigma)
        teacher_hidden = teacher.get_hidden_states()
    
    student_output = student(pixel_values, z, sigma)
    student_hidden = student.get_hidden_states()
    
    # Distillation loss
    loss = F.mse_loss(student_output, teacher_output)
    loss += feature_weight * F.mse_loss(student_hidden, teacher_hidden)
    
    # Task loss (optional)
    loss += task_weight * cross_entropy(student_output, labels)
```

## Benefits

| Benefit | Description |
|---|---|
| **Compression** | N blocks → M blocks (M < N) |
| **Faster inference** | Fewer blocks to evaluate |
| **Better quality** | Teacher provides richer signal than labels |
| **Flexible** | Can compress to any number of blocks |

## Configuration

```bash
# Train teacher first
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --num_blocks 6 --num_epochs 1500

# Distill to student
uv run python -m diffusionblocks.main train cifar100 \
    --model_type dblock --num_blocks 3 \
    --distill --teacher_path logs/teacher/last.ckpt \
    --distill_weight 0.5 --task_weight 0.5
```

## References

- Distilling the Knowledge in a Neural Network (Hinton et al., 2015)
- QLoRA (Dettmers et al., 2023)
- Original DiffusionBlocks paper (Shing et al., 2026)
