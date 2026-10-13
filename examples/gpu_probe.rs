//! GPU bring-up probe: does the `wgpu` backend actually work on this
//! machine, and is it faster than the NdArray CPU backend on the workloads
//! the crate cares about? A measurement, not a promise.
//!
//! ```text
//! cargo run --release --features wgpu --example gpu_probe
//! cargo run --release --example gpu_probe          # CPU half only
//! ```
//!
//! Three workloads, timed identically on each backend with a device sync per
//! repetition so the wall time is the operation's, not the queue's:
//!
//! - one f32 matmul at 5120x5120x5120 (268 GFLOP), the shape that decides
//!   whether a big trunk step is worth offloading;
//! - one forward of the tiny test-scale `GeometricStudent` (hidden 64, 4
//!   layers, 4x32 tokens);
//! - one forward of the default local student (hidden 512, 8 layers, 4x64
//!   tokens) — the config this machine actually trains.
//!
//! After the timings, a cross-check runs the *same* inputs through both
//! backends (identical weights moved over by record) and reports the max
//! divergence: a faster answer that is a different answer is not a result.
//!
//! Every number is labeled with the backend that produced it. A failed
//! adapter init is reported with the exact error and counted as the result:
//! "wgpu does not start on this machine" is a finding, not a crash.

#[cfg(feature = "wgpu")]
use anyhow::Result;

#[cfg(feature = "wgpu")]
mod probe {
    use anyhow::{anyhow, Context, Result};
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::wgpu::graphics::AutoGraphicsApi;
    use burn::backend::wgpu::{init_setup, RuntimeOptions, WgpuDevice};
    use burn::backend::{NdArray, Wgpu};
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder};
    use burn::tensor::backend::Backend;
    use burn::tensor::{Distribution, Int, Tensor};
    use diffusionblocks::student::{GeometricStudent, StudentConfig};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::time::Instant;

    /// One timed row of the report: per-repetition wall times in
    /// milliseconds, device-synchronized.
    struct Row {
        label: &'static str,
        times_ms: Vec<f64>,
    }

    impl Row {
        fn min(&self) -> f64 {
            self.times_ms.iter().copied().fold(f64::INFINITY, f64::min)
        }

        fn mean(&self) -> f64 {
            self.times_ms.iter().sum::<f64>() / self.times_ms.len() as f64
        }
    }

    fn sync<B: Backend>(device: &B::Device, what: &str) -> Result<()> {
        B::sync(device).map_err(|err| anyhow!("{what}: device sync failed: {err}"))
    }

    /// Time `op` on `device`: `warmup` untimed calls (shader compilation and
    /// lazy buffer setup land here), then `reps` synchronized, timed calls.
    fn bench<B: Backend>(
        label: &'static str,
        device: &B::Device,
        warmup: usize,
        reps: usize,
        mut op: impl FnMut() -> Result<()>,
    ) -> Result<Row> {
        for _ in 0..warmup {
            op().with_context(|| format!("{label}: warmup"))?;
            sync::<B>(device, label)?;
        }
        let mut times_ms = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = Instant::now();
            op().with_context(|| format!("{label}: timed repetition"))?;
            sync::<B>(device, label)?;
            times_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        Ok(Row { label, times_ms })
    }

    /// 5120^3 f32 matmul: the stand-in for one dense trunk projection at
    /// scale. 2 * 5120^3 = 268.4 GFLOP per call.
    fn matmul_row<B: Backend<FloatElem = f32>>(device: &B::Device) -> Result<Row> {
        const N: usize = 5120;
        let a = Tensor::<B, 2>::random([N, N], Distribution::Default, device);
        let b = Tensor::<B, 2>::random([N, N], Distribution::Default, device);
        bench::<B>("matmul 5120^3 f32", device, 1, 5, move || {
            let c = a.clone().matmul(b.clone());
            std::hint::black_box(&c);
            Ok(())
        })
    }

    /// The tiny test-scale student config, copied from
    /// `student.rs`'s `tiny_config`.
    fn tiny_student_config() -> StudentConfig {
        StudentConfig {
            hidden_size: 64,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 2,
            context: 32,
            intermediate_size: 32,
            cond_hidden_size: 16,
            frequency_embedding_size: 16,
            experts_per_box: 2,
            geom_slots: 4,
            geom_dim: 8,
            geom_refine_steps: 2,
            geom_cond_hidden: 16,
            geom_frequency: 8,
            ..StudentConfig::default()
        }
    }

    /// One inference forward of a student, batch `b` x `n` tokens.
    fn student_row<B: Backend<FloatElem = f32>>(
        device: &B::Device,
        config: &StudentConfig,
        label: &'static str,
        b: usize,
        n: usize,
    ) -> Result<Row> {
        B::seed(device, 42);
        let student =
            GeometricStudent::<B>::new(config, device).with_context(|| format!("build {label}"))?;
        let tokens = tokens::<B>(device, b, n);
        bench::<B>(label, device, 2, 10, move || {
            let out = student.forward(tokens.clone()).context("student forward")?;
            std::hint::black_box(&out.logits);
            Ok(())
        })
    }

    /// Deterministic token ids, valid for any vocab >= 256.
    fn tokens<B: Backend>(device: &B::Device, b: usize, n: usize) -> Tensor<B, 2, Int> {
        let ids: Vec<i64> = (0..b * n).map(|i| ((i * 7 + 3) % 250 + 4) as i64).collect();
        Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device).reshape([b, n])
    }

    fn print_rows(backend: &str, rows: &[Row]) {
        for row in rows {
            println!(
                "  {backend:32} {:26} min {:>9.2} ms   mean {:>9.2} ms",
                row.label,
                row.min(),
                row.mean()
            );
        }
    }

    /// Max absolute and max relative difference between two same-shaped
    /// f32 result sets. Relative difference is against the CPU magnitude,
    /// floored at 1e-6 so near-zero entries cannot blow the ratio up.
    fn max_diffs(gpu: Vec<f32>, cpu: Vec<f32>, what: &str) -> Result<(f64, f64)> {
        anyhow::ensure!(
            gpu.len() == cpu.len(),
            "{what}: backends returned different element counts ({} vs {})",
            gpu.len(),
            cpu.len()
        );
        let mut abs = 0f64;
        let mut rel = 0f64;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            let d = (*g as f64 - *c as f64).abs();
            abs = abs.max(d);
            rel = rel.max(d / (*c as f64).abs().max(1e-6));
        }
        Ok((abs, rel))
    }

    /// The honesty check: identical inputs and weights on both backends must
    /// produce the same numbers up to float reduction order, on the matmul
    /// and on the tiny student's logits.
    fn cross_check(cpu_device: &NdArrayDevice, gpu_device: &WgpuDevice) -> Result<()> {
        NdArray::<f32>::seed(cpu_device, 7);
        let a_cpu =
            Tensor::<NdArray<f32>, 2>::random([1024, 1024], Distribution::Default, cpu_device);
        let b_cpu =
            Tensor::<NdArray<f32>, 2>::random([1024, 1024], Distribution::Default, cpu_device);
        let c_cpu = a_cpu.clone().matmul(b_cpu.clone());
        let a_gpu =
            Tensor::<Wgpu<f32>, 2>::from_data(a_cpu.into_data().convert::<f32>(), gpu_device);
        let b_gpu =
            Tensor::<Wgpu<f32>, 2>::from_data(b_cpu.into_data().convert::<f32>(), gpu_device);
        let c_gpu = a_gpu.matmul(b_gpu);
        let cpu_vals: Vec<f32> = c_cpu.into_data().convert::<f32>().iter::<f32>().collect();
        let gpu_vals: Vec<f32> = c_gpu.into_data().convert::<f32>().iter::<f32>().collect();
        let (abs, rel) = max_diffs(gpu_vals, cpu_vals, "matmul 1024^3")?;
        println!("  matmul 1024^3 f32        max abs diff {abs:.3e}   max rel diff {rel:.3e}");

        // Identical weights on both backends: build on CPU, forward there,
        // then move the weights over through an in-memory recorder round-trip
        // and forward again.
        NdArray::<f32>::seed(cpu_device, 42);
        let student_cpu = GeometricStudent::<NdArray<f32>>::new(&tiny_student_config(), cpu_device)
            .context("build the tiny student on CPU")?;
        let logits_cpu = student_cpu
            .forward(tokens::<NdArray<f32>>(cpu_device, 2, 32))
            .context("CPU student forward")?
            .logits;
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::new();
        let bytes = recorder
            .record(student_cpu.into_record(), ())
            .map_err(|err| anyhow!("serialize the CPU student record: {err}"))?;
        let student_gpu = GeometricStudent::<Wgpu<f32>>::new(&tiny_student_config(), gpu_device)
            .context("build the tiny student on GPU")?
            .load_record(
                recorder
                    .load(bytes, gpu_device)
                    .map_err(|err| anyhow!("load the student record on the GPU: {err}"))?,
            );
        let logits_gpu = student_gpu
            .forward(tokens::<Wgpu<f32>>(gpu_device, 2, 32))
            .context("GPU student forward")?
            .logits;
        let cpu_vals: Vec<f32> = logits_cpu
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        let gpu_vals: Vec<f32> = logits_gpu
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        let (abs, rel) = max_diffs(gpu_vals, cpu_vals, "tiny student logits")?;
        println!("  tiny student logits      max abs diff {abs:.3e}   max rel diff {rel:.3e}");
        println!(
            "  (reduction order differs between backends; small nonzero diffs are \
             expected, O(1) diffs would mean the GPU computes something else)"
        );
        Ok(())
    }

    pub(crate) fn run() -> Result<()> {
        println!("gpu probe: same workloads, both backends, wall time per op (synced)\n");

        let cpu_device = NdArrayDevice::default();
        let cpu_rows = [
            matmul_row::<NdArray<f32>>(&cpu_device)?,
            student_row::<NdArray<f32>>(
                &cpu_device,
                &tiny_student_config(),
                "student tiny fwd 4x32",
                4,
                32,
            )?,
            student_row::<NdArray<f32>>(
                &cpu_device,
                &StudentConfig::default(),
                "student default fwd 4x64",
                4,
                64,
            )?,
        ];
        print_rows(
            &format!("ndarray ({})", NdArray::<f32>::name(&cpu_device)),
            &cpu_rows,
        );

        // cubecl panics (not errs) when no adapter exists; catch it so the
        // exact failure is reported as the probe's finding, then bail loudly.
        let setup = catch_unwind(AssertUnwindSafe(|| {
            init_setup::<AutoGraphicsApi>(&WgpuDevice::default(), RuntimeOptions::default())
        }));
        let setup = match setup {
            Ok(setup) => setup,
            Err(payload) => {
                let detail = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "non-string panic payload".to_string());
                println!("\nwgpu adapter init FAILED: {detail}");
                println!("CPU numbers above stand; the GPU column does not exist here.");
                return Err(anyhow!("wgpu adapter init failed: {detail}"));
            }
        };
        let info = setup.adapter.get_info();
        println!("\nwgpu adapter: {}", info.name);
        println!(
            "  backend {:?}  device_type {:?}  driver {} {}",
            info.backend, info.device_type, info.driver, info.driver_info
        );

        let gpu_device = WgpuDevice::default();
        let gpu_rows = [
            matmul_row::<Wgpu<f32>>(&gpu_device)?,
            student_row::<Wgpu<f32>>(
                &gpu_device,
                &tiny_student_config(),
                "student tiny fwd 4x32",
                4,
                32,
            )?,
            student_row::<Wgpu<f32>>(
                &gpu_device,
                &StudentConfig::default(),
                "student default fwd 4x64",
                4,
                64,
            )?,
        ];
        print_rows(
            &format!("wgpu ({})", Wgpu::<f32>::name(&gpu_device)),
            &gpu_rows,
        );

        println!("\nspeedup wgpu vs ndarray (min wall time):");
        for (cpu, gpu) in cpu_rows.iter().zip(gpu_rows.iter()) {
            println!("  {:26} {:.2}x", cpu.label, cpu.min() / gpu.min().max(1e-9));
        }

        println!("\ncross-backend check (same inputs, same weights):");
        cross_check(&cpu_device, &gpu_device)?;

        println!(
            "\nnote: `min` is the honest number here -- it is the repetition least \
             disturbed by the OS, and shader compilation sits in the warmup."
        );
        Ok(())
    }
}

#[cfg(feature = "wgpu")]
fn main() -> Result<()> {
    probe::run()
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    println!(
        "gpu_probe was built without the `wgpu` feature: there is no GPU backend to probe. \
         Rebuild with --features wgpu."
    );
}
