# Precision Denoising & I/O Uring

Advanced performance features for DiffusionBlocks++.

> **In this repository.** Two independent modules.
>
> **Precision** — `src/precision.rs`. The `ndarray` backend computes in f32
> only, so `Precision::{Bf16, F16}` **emulate** the formats: values are rounded
> onto the target grid with correct round-to-nearest-even, subnormals and
> overflow, while arithmetic stays in f32.
>
> This models representation error exactly but not a real low-precision
> kernel's accumulation order, and it is **slower**, not faster. It is a
> numerical-analysis tool — it answers "how much accuracy would bf16 cost
> here?" — and `Precision::round_scalar` is the single place a native cast
> swaps in once a backend offers one.
>
> `PrecisionPolicy::mixed(Bf16, switch_sigma)` runs high-sigma windows coarse
> and low-sigma windows in f32: at high sigma the latent is dominated by noise
> of magnitude sigma, so a relative error of `2^-8` sits far below the noise
> floor. Certified: relative error `<= 2^-p`, and rounding is idempotent, so
> the output really lies on the target grid. Verified against an independent
> bit-level bf16 implementation.
>
> CLI: `--precision bf16 --precision-switch 1.0`.
>
> **I/O** — `src/rawdata.rs`. `StreamingSplit` uses positional reads (`pread`,
> one syscall instead of `seek` + `read`), sorts sampled indices and
> **coalesces contiguous runs into a single read**, and lands bytes in reusable
> buffers so steady-state batching allocates nothing. `reads_issued()` exposes
> the syscall count, and the label scan walks the file in 4 MiB chunks rather
> than one header read per record.
>
> Native `io_uring` submission would need an external crate and is deliberately
> not vendored: the measurable win it targets is what run coalescing already
> delivers.
>
> **Profiling** — `src/profile.rs`. Named scopes with exact (not estimated)
> percentiles, ranked by total time. Percentiles rather than means alone,
> because a step that is usually fast but occasionally stalls has a healthy
> mean and a terrible p95.


## Precision Denoising

### Overview

Use different numerical precision for different blocks or noise regimes
to optimize the tradeoff between quality and memory/speed.

### Precision Strategy

| Noise Regime | Precision | Reason |
|---|---|---|
| High noise (σ > 10) | FP32 | Large gradients need precision for stability |
| Medium noise (1 < σ < 10) | BF16 | Balanced precision and memory |
| Low noise (σ < 1) | FP16 | Fine refinement, less precision needed |
| Inference | FP16/INT8 | Maximum speed |

### Dynamic Precision Scaling

```python
class PrecisionDenoiser:
    """Apply different precision per block and noise level."""
    
    def __init__(self, model, config):
        self.model = model
        self.strategy = config.get('precision_strategy', 'dynamic')
        self.high_noise_dtype = config.get('high_noise_dtype', torch.float32)
        self.low_noise_dtype = config.get('low_noise_dtype', torch.bfloat16)
    
    def denoise(self, x, sigma, block_idx):
        """Denoise with appropriate precision."""
        dtype = self.get_dtype(block_idx, sigma)
        
        with torch.cuda.amp.autocast(dtype=dtype):
            output = self.model.blocks[block_idx](x, sigma)
        
        return output
    
    def get_dtype(self, block_idx, sigma):
        """Determine precision for current block and noise level."""
        if self.strategy == 'fp32':
            return torch.float32
        elif self.strategy == 'fp16':
            return torch.float16
        elif self.strategy == 'bf16':
            return torch.bfloat16
        elif self.strategy == 'dynamic':
            sigma_val = sigma.mean().item()
            if sigma_val > 10.0:
                return self.high_noise_dtype
            elif sigma_val > 1.0:
                return torch.bfloat16
            else:
                return self.low_noise_dtype
        else:
            return torch.float32
```

### Memory Savings

| Strategy | Memory per Block | Quality Impact |
|---|---|---|
| All FP32 | 4 bytes/param | Baseline |
| All BF16 | 2 bytes/param | Minimal loss |
| All FP16 | 2 bytes/param | Small loss |
| Dynamic | 2-4 bytes/param | Minimal loss |

## I/O Uring Support

### Overview

Linux io_uring provides high-performance asynchronous I/O for:

- **Async Data Loading**: Non-blocking file reads
- **Async Checkpointing**: Non-blocking checkpoint saves
- **Batched Syscalls**: Reduce syscall overhead
- **Zero-Copy**: Direct data transfer

### Why io_uring?

| Feature | Standard I/O | io_uring |
|---|---|---|
| Blocking | Yes | No |
| Async | Limited | Full |
| Batch syscalls | No | Yes |
| Zero-copy | No | Yes |
| Polling | No | Yes |
| Overhead | High (syscalls) | Low (shared ring) |

### Integration with Data Loading

```python
class IOUringDataLoader:
    """High-performance data loader using Linux io_uring."""
    
    def __init__(self, data_dir, queue_depth=128):
        self.data_dir = data_dir
        self.queue_depth = queue_depth
        self.ring = self._init_io_uring(queue_depth)
    
    def _init_io_uring(self, queue_depth):
        """Initialize io_uring instance."""
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

### Async Checkpoint Saving

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

### Performance Profiler

```python
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
# Precision denoising configuration
precision:
  strategy: dynamic  # fp32, fp16, bf16, dynamic
  high_noise_dtype: fp32
  low_noise_dtype: bf16

# I/O uring configuration
io_uring:
  enabled: true
  queue_depth: 128
  async_checkpoint: true
  async_data_loading: true
  polling_mode: true

# Profiler configuration
profiling:
  enabled: false
  log_every_n_steps: 100
  profile_memory: true
  profile_compute: true
```

## References

- io_uring (Axboe, 2019)
- PyTorch AMP (Automatic Mixed Precision)
- Original DiffusionBlocks paper (Shing et al., 2026)

---

See also: [Quality Gate](Quality-Gate.md) · [Training Guide](Training-Guide.md) · [Inference Guide](Inference-Guide.md) · [Home](Home.md)
