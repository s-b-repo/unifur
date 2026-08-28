# Multi-Block Denoising

Training and inference with multiple blocks simultaneously, including
parallel trajectories, hybrid strategies, and adaptive depth control.

> **In this repository.** Implemented in `src/multi_block.rs` as `Strategy`
> (`Sequential` / `Parallel{k}` / `Hybrid{k, warmup_frac}` /
> `Adaptive{k_max, conf_threshold}`) driven through the shared
> `solver::SolverState`, so **any strategy composes with any solver** — the
> integration test exercises the full 4x5 matrix. Alongside them,
> `sample_planned` + `PlannedConfig` replace the fixed schedule with a search
> (see [Planned Denoising](#planned-denoising)).
>
> `SamplingStats` reports `model_calls`, `layers_executed`, `mean_span_width`,
> `spans`, `planning_calls`/`planning_layers` and a per-block gate ledger.
> `layers_executed` is the honest cost measure: `parallel-2` makes the same
> number of model calls as `sequential` but runs nearly twice the layers. It
> also carries a solver's corrector evaluations and any planning work, which is
> why `mean_span_width` averages the recorded `spans` instead — otherwise Heun's
> spans would be reported at twice their real width.
>
> `Strategy::Adaptive` widens the span while confidence is below the threshold
> **and narrows it again** once the estimate is confident, so the extra depth
> is spent only where it is needed.
>
> CLI: `dblocks sample --strategy ... --k N`, `dblocks bench`.
> Precision denoising is `MultiBlockConfig::precision` — see
> [Precision & I/O](Precision-IO.md). Quality gating is
> [Quality Gate](Quality-Gate.md).


## Overview

Multi-block denoising encompasses several strategies for training and
inference with multiple blocks at once:

| Strategy | Description | Use Case |
|---|---|---|
| **Sequential** | One block at a time (original) | Baseline, memory-constrained |
| **Parallel** | K blocks simultaneously | Faster training |
| **Hybrid** | Mix of sequential and parallel | Flexible tradeoffs |
| **Adaptive** | Dynamic block skipping | Dynamic compute |
| **Quality-Gated** | Per-step quality checks | Prevent bad denoising |

## Sequential Denoising (Original)

The baseline approach from the original DiffusionBlocks paper.

```
Step 0: Block 0 (σ_max → σ_a)
Step 1: Block 1 (σ_a → σ_b)
Step 2: Block 2 (σ_b → σ_min)
```

**Pros**: Simple, minimal memory, proven convergence
**Cons**: Slow (B× more steps), no inter-block communication

## Parallel Denoising

Train K blocks simultaneously on overlapping noise windows.

```
Step 0: Block 0 + Block 1 (overlap on σ_a)
Step 1: Block 1 + Block 2 (overlap on σ_b)
Step 2: Block 0 + Block 2 (cross-fork)
```

**Pros**: K× faster, inter-block cooperation via overlap
**Cons**: More memory (K blocks), needs consistency loss

## Hybrid Denoising

Mix sequential and parallel strategies based on training phase or
noise regime.

### Phase-Based Hybrid

```python
if epoch < warmup_epochs:
    # Start with sequential for stability
    train_sequential()
elif epoch < mid_epochs:
    # Switch to parallel for speed
    train_parallel(K=2)
else:
    # Full parallel with cross-fork
    train_parallel(K=3, cross_fork=True)
```

### Noise-Regime Hybrid

Different noise regimes use different strategies:

```
High noise (σ > 10):  Parallel K=3 (needs cooperation)
Medium noise (1 < σ < 10): Parallel K=2
Low noise (σ < 1):    Sequential (fine refinement)
```

### Block-Aware Hybrid

```python
# Early blocks (high noise): parallel
# Late blocks (low noise): sequential
for i, block in enumerate(blocks):
    if i < num_blocks // 2:
        # Parallel training
        train_block_parallel(block, K=2)
    else:
        # Sequential training
        train_block_sequential(block)
```

## Adaptive Denoising

Dynamically adjust the number of blocks and training strategy based on
model confidence.

### Confidence-Based Block Selection

```python
# Compute confidence for each block
confidences = []
for block in blocks:
    output = block(x, sigma)
    confidence = compute_confidence(output)
    confidences.append(confidence)

# Select blocks with low confidence (need more training)
active_blocks = [i for i, c in enumerate(confidences) if c < threshold]
```

### Dynamic K Adjustment

```python
# Start with K=1, increase as training progresses
K = min(max_K, 1 + epoch // ramp_epochs)

# Or: adjust based on loss
if loss_plateau:
    K = min(max_K, K + 1)  # Add more blocks
elif loss_unstable:
    K = max(1, K - 1)      # Reduce to stabilize
```

## Planned Denoising

`Strategy::Adaptive` reacts: it widens the span after seeing an unconfident
estimate and narrows it after a confident one, with no view of what comes next.
Planned sampling replaces the reaction with a search.

```bash
dblocks sample --planned --plan-depth 2 --plan-beam 3 --plan-budget 32
```

```rust
let (logits, stats, trace) = model.sample_planned(&pixels, &PlannedConfig {
    budget: Budget { max_evaluations: 48, max_depth: 2, beam_width: 3 },
    ..PlannedConfig::default()
}, &mut rng);
```

Each step scores candidate `(sigma, span)` pairs and, when the depth allows,
rolls the promising ones forward — then commits **only the winner's first
step** and re-plans. Depth 0 is certified to be exactly the greedy policy, so
this is a generalization of the adaptive strategy rather than a replacement for
it.

The sigma is chosen too, not just the span, which is the part `Adaptive` cannot
do: the schedule is no longer fixed before the first model call. That needs a
progress term in log-sigma to work at all — the most accurate single step is
always the shortest one, so without a reward for descending the planner would
never reach `sigma_min`.

Planning is not free: `PlanTrace` reports the evaluations spent per committed
step and `SamplingStats::planning_overhead` the fraction of executed layers that
went to planning rather than sampling. Full detail:
[Next-Step Planning](Next-Step-Planning.md).

## Quality-Gated Denoising

Prevent bad denoising at each step by checking quality before accepting
the output.

### Quality Metrics

1. **MSE Check**: Compare denoised output to target
2. **Cosine Similarity**: Check direction alignment
3. **Confidence Threshold**: Minimum confidence to accept
4. **Gradient Norm**: Reject steps with exploding gradients

### Quality Gate Implementation

```python
class QualityGate:
    def __init__(self, mse_threshold=0.1, cos_threshold=0.9):
        self.mse_threshold = mse_threshold
        self.cos_threshold = cos_threshold
    
    def check(self, denoised, target, sigma):
        # MSE check
        mse = F.mse_loss(denoised, target, reduction='mean')
        if mse > self.mse_threshold * sigma:
            return False, f"MSE too high: {mse:.4f}"
        
        # Cosine similarity check
        cos_sim = F.cosine_similarity(
            denoised.flatten(1), target.flatten(1)
        ).mean()
        if cos_sim < self.cos_threshold:
            return False, f"Cosine similarity too low: {cos_sim:.4f}"
        
        return True, "OK"
    
    def filter_batch(self, denoised, target, sigma):
        """Filter out bad samples from batch."""
        batch_size = denoised.shape[0]
        mask = torch.ones(batch_size, dtype=torch.bool)
        
        for i in range(batch_size):
            ok, msg = self.check(denoised[i], target[i], sigma[i])
            if not ok:
                mask[i] = False
        
        return mask
```

### Per-Layer Quality Gate

Each block has its own quality gate:

```python
class QualityGatedBlock(nn.Module):
    def __init__(self, block, quality_gate):
        super().__init__()
        self.block = block
        self.quality_gate = quality_gate
        self.prev_output = None
    
    def forward(self, x, sigma, target=None):
        output = self.block(x, sigma)
        
        if target is not None and self.training:
            ok, msg = self.quality_gate.check(output, target, sigma)
            if not ok:
                # Use previous output or skip update
                if self.prev_output is not None:
                    output = self.prev_output
                else:
                    output = x  # Identity fallback
        
        self.prev_output = output.detach()
        return output
```

## Precision Denoising

Use different numerical precision for different blocks or noise regimes.

### Mixed Precision Strategy

```python
# High noise: FP32 (needs precision for large gradients)
# Low noise: BF16 (fine refinement, less precision needed)
# Inference: FP16 or INT8

def get_precision_for_block(block_idx, num_blocks, noise_level):
    if noise_level > 10.0:
        return torch.float32
    elif noise_level > 1.0:
        return torch.bfloat16
    else:
        return torch.float16
```

### Dynamic Precision Scaling

```python
class PrecisionDenoiser:
    def __init__(self, model):
        self.model = model
        self.scaler = torch.cuda.amp.GradScaler()
    
    def denoise(self, x, sigma, block_idx):
        precision = get_precision_for_block(block_idx, sigma)
        
        with torch.cuda.amp.autocast(dtype=precision):
            output = self.model.blocks[block_idx](x, sigma)
        
        return output
```

## Hybrid Loop Graph Dynamic Transformers

Dynamic computation graph that adapts to what needs to be done at each step.

### Dynamic Graph Construction

```python
class HybridLoopGraph:
    """Dynamic computation graph that adapts per sample."""
    
    def __init__(self, blocks, router, quality_gate):
        self.blocks = blocks
        self.router = router
        self.quality_gate = quality_gate
    
    def forward(self, x, sigma, mode='adaptive'):
        if mode == 'sequential':
            return self._sequential(x, sigma)
        elif mode == 'parallel':
            return self._parallel(x, sigma)
        elif mode == 'adaptive':
            return self._adaptive(x, sigma)
        elif mode == 'hybrid':
            return self._hybrid(x, sigma)
    
    def _adaptive(self, x, sigma):
        """Dynamically choose which blocks to run."""
        outputs = []
        confidences = []
        
        for i, block in enumerate(self.blocks):
            output = block(x, sigma)
            confidence = self.compute_confidence(output)
            
            if confidence < self.threshold:
                outputs.append(output)
                confidences.append(confidence)
            
            # Early exit if confident
            if confidence > self.exit_threshold:
                break
        
        if not outputs:
            return x
        
        # Weighted combination
        weights = F.softmax(torch.tensor(confidences), dim=0)
        return sum(w * o for w, o in zip(weights, outputs))
    
    def _hybrid(self, x, sigma):
        """Mix sequential and parallel based on noise level."""
        if sigma > 10.0:
            # High noise: parallel for cooperation
            return self._parallel(x, sigma)
        else:
            # Low noise: sequential for refinement
            return self._sequential(x, sigma)
```

### Loop Graph with Skip Connections

```python
class LoopedGraphTransformer(nn.Module):
    """Transformer with dynamic loop graph."""
    
    def __init__(self, num_blocks, hidden_size, max_iterations=10):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size) for _ in range(num_blocks)
        ])
        self.halting = nn.Linear(hidden_size, 1)
        self.max_iterations = max_iterations
    
    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        
        # Halting probabilities
        halting_probs = torch.zeros(batch_size, device=device)
        outputs = torch.zeros_like(x)
        
        for iteration in range(self.max_iterations):
            # Compute halting probability
            h = self.halting(x.mean(dim=1))
            p = torch.sigmoid(h).squeeze(-1)
            
            # Update outputs for non-halted samples
            active = halting_probs < 0.99
            if not active.any():
                break
            
            # Run active blocks
            block_idx = iteration % len(self.blocks)
            block_output = self.blocks[block_idx](x)
            
            # Accumulate weighted outputs
            weight = (1 - halting_probs) * p
            outputs = outputs + weight.unsqueeze(1).unsqueeze(1) * block_output
            halting_probs = halting_probs + weight
        
        return outputs
```

## I/O Uring Support

Linux io_uring for high-performance async I/O during training.

### Why io_uring?

- **Async I/O**: Non-blocking file reads/writes
- **Batched syscalls**: Reduce syscall overhead
- **Zero-copy**: Direct data transfer
- **Polling**: Busy-polling for low latency

### Integration with Data Loading

```python
import ctypes
import os

class IOUringDataLoader:
    """High-performance data loader using Linux io_uring."""
    
    def __init__(self, data_dir, queue_depth=128):
        self.data_dir = data_dir
        self.queue_depth = queue_depth
        self.ring = self._init_io_uring(queue_depth)
    
    def _init_io_uring(self, queue_depth):
        """Initialize io_uring instance."""
        # Requires liburing or similar
        # This is a simplified interface
        try:
            import liburing
            ring = liburing.io_uring()
            liburing.io_uring_queue_init(queue_depth, ring, 0)
            return ring
        except ImportError:
            return None
    
    def read_batch_async(self, file_paths):
        """Read multiple files asynchronously."""
        if self.ring is None:
            # Fallback to synchronous
            return [self._read_file(p) for p in file_paths]
        
        # Submit read requests
        results = []
        for path in file_paths:
            fd = os.open(path, os.O_RDONLY)
            buf = bytearray(os.path.getsize(path))
            
            # Prepare read request
            sqe = liburing.io_uring_get_sqe(self.ring)
            liburing.io_uring_prep_read(sqe, fd, buf, len(buf), 0)
            liburing.io_uring_submit(self.ring)
            results.append((fd, buf))
        
        # Collect completions
        outputs = []
        for fd, buf in results:
            cqe = liburing.io_uring_wait_cqe(self.ring)
            outputs.append(bytes(buf))
            os.close(fd)
            liburing.io_uring_cqe_seen(self.ring, cqe)
        
        return outputs
```

### Checkpoint Saving with io_uring

```python
class AsyncCheckpointSaver:
    """Save checkpoints asynchronously using io_uring."""
    
    def __init__(self, save_dir, use_io_uring=True):
        self.save_dir = save_dir
        self.use_io_uring = use_io_uring
        self.pending_saves = {}
    
    def save_async(self, state_dict, filename):
        """Save checkpoint asynchronously."""
        path = os.path.join(self.save_dir, filename)
        
        if self.use_io_uring:
            # Serialize to buffer
            import io
            buf = io.BytesIO()
            torch.save(state_dict, buf)
            data = buf.getvalue()
            
            # Async write via io_uring
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            # ... io_uring write submission
            self.pending_saves[fd] = path
        else:
            # Synchronous fallback
            torch.save(state_dict, path)
    
    def wait_for_saves(self):
        """Wait for all pending saves to complete."""
        for fd, path in self.pending_saves.items():
            # Wait for completion
            pass
        self.pending_saves.clear()
```

## Code Quality Features

### Static Analysis Integration

```yaml
# .github/workflows/quality.yml
name: Code Quality
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Ruff
        run: ruff check src/
      - name: Run MyPy
        run: mypy src/diffusionblocks/
      - name: Run Tests
        run: pytest tests/ -v --cov
```

### Performance Profiling

```python
# Built-in profiling support
class Profiler:
    """Profile denoising steps for performance analysis."""
    
    def __init__(self):
        self.step_times = []
        self.block_times = {}
        self.memory_usage = []
    
    def profile_step(self, block_idx, sigma, fn):
        """Profile a single denoising step."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        result = fn()
        end.record()
        
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        
        if block_idx not in self.block_times:
            self.block_times[block_idx] = []
        self.block_times[block_idx].append(elapsed)
        
        return result
    
    def get_summary(self):
        """Get performance summary."""
        summary = {}
        for block_idx, times in self.block_times.items():
            summary[f"block_{block_idx}"] = {
                "mean_ms": sum(times) / len(times),
                "max_ms": max(times),
                "min_ms": min(times),
            }
        return summary
```

## Configuration

```yaml
# configs/multi_block.yaml
multi_block:
  strategy: hybrid  # sequential, parallel, hybrid, adaptive
  parallel_k: 2
  cross_fork: true
  consistency_loss: true
  consistency_weight: 0.1
  
quality_gate:
  enabled: true
  mse_threshold: 0.1
  cos_threshold: 0.9
  min_confidence: 0.5
  
adaptive:
  enabled: true
  target_depth: 2.0
  max_halting_prob: 0.5
  ramp_epochs: 100
  
precision:
  strategy: dynamic  # fp32, fp16, bf16, dynamic
  high_noise_dtype: fp32
  low_noise_dtype: bf16
  
io_uring:
  enabled: true
  queue_depth: 128
  async_checkpoint: true
  
profiling:
  enabled: false
  log_every_n_steps: 100
```

## References

- Original DiffusionBlocks paper (Shing et al., 2026)
- Universal Transformers (Dehghani et al., 2019)
- PonderNet (Banino et al., 2021)
- Looped Transformers (Fan et al., 2025)
- io_uring (Axboe, 2019)
