//! Native CUDA backend: hand-written kernels (src/cuda/jsv.cu) driven through
//! cuRAND, loaded as PTX and launched via the CUDA driver API.
//!
//! This exists alongside the CubeCL backend rather than replacing it. CubeCL
//! cannot call cuRAND - kernel bodies are transpiled, so an external device
//! library is out of reach - and cuRAND is what makes the generator a runtime
//! choice rather than something compiled in.

use anyhow::{Context, Result, anyhow};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig,
                     PushKernelArg};
use std::sync::Arc;
use tracing::info;

use crate::gpu_markov_chain::{CurandRng, InitialChains, JsvDevice};

/// PTX for all three generators, compiled by build.rs and embedded so there is
/// no runtime file dependency.
const JSV_PTX: &str = include_str!(env!("PERMANENT_JSV_PTX"));

/// Threads per block. One warp, matching the CubeCL cube size. Each thread
/// runs one chain as a long serial loop, so total threads equal the chain
/// count and the only thing that matters is spreading blocks across SMs: at
/// 2048 chains a 256-thread block would leave a 128-SM device running 8
/// blocks. Small blocks also keep divergent chains from stalling each other.
const BLOCK: u32 = 32;

/// Shared memory the staged variants need: row and col matchings for every
/// chain in a block. Graphs past this budget fall back to the `_global`
/// kernels, which read the matchings straight out of global memory.
fn shared_bytes(size: usize) -> usize {
    2 * size * BLOCK as usize * std::mem::size_of::<u32>()
}

/// Advice appended to every failure to bring up the native CUDA backend.
const FALLBACK_HINT: &str = "Use `--backend cubecl_philox` for the portable CubeCL/Vulkan path, or `--backend cpu`.";

/// Open a CUDA context, turning both failure modes into ordinary errors.
///
/// cudarc *panics* rather than returning `Err` when it cannot dlopen
/// libcuda.so - which is exactly the "this machine has no NVIDIA driver" case
/// the CLI has to report cleanly - so the first driver touch happens inside
/// `catch_unwind`, with the panic hook silenced so the backtrace never reaches
/// the user. This runs during single-threaded start-up, so replacing the
/// global hook briefly is safe.
fn open_context(ordinal: usize) -> Result<Arc<CudaContext>> {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let attempt = std::panic::catch_unwind(|| CudaContext::new(ordinal));
    std::panic::set_hook(previous_hook);

    match attempt {
        Ok(Ok(ctx)) => Ok(ctx),
        Ok(Err(error)) => Err(anyhow!(
            "could not initialise CUDA device {ordinal}: {error}. The native CUDA \
             backend needs an NVIDIA driver (libcuda.so) and a visible GPU. \
             {FALLBACK_HINT}"
        )),
        Err(_) => Err(anyhow!(
            "no usable NVIDIA driver on this machine: libcuda.so could not be \
             loaded. The native CUDA backend needs the NVIDIA driver installed \
             (the CUDA toolkit alone is not enough - libcuda.so ships with the \
             driver). {FALLBACK_HINT}"
        )),
    }
}

pub struct NativeCudaDevice {
    size: usize,
    num_chains: usize,
    rng: CurandRng,
    stream: Arc<CudaStream>,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    warmup_fn: CudaFunction,
    occupancy_fn: CudaFunction,
    ratio_fn: CudaFunction,

    row_match: CudaSlice<u32>,
    col_match: CudaSlice<u32>,
    holes_u: CudaSlice<u32>,
    holes_v: CudaSlice<u32>,
    adjacency: CudaSlice<u32>,
    active_counts: CudaSlice<u32>,
    /// Opaque cuRAND states, `state_size * num_chains` bytes.
    rng_states: CudaSlice<u8>,

    // Every buffer below is allocated once and reused. Sizes are fixed by the
    // graph and chain count, and reallocating (or worse, cloning, which
    // device-copies) per pass showed up directly in step time.
    shared_bytes: usize,
    weights: CudaSlice<f32>,
    exp_beta: CudaSlice<f32>,
    next_weights: CudaSlice<f32>,
    ratio_terms: CudaSlice<f32>,
    sums: CudaSlice<f32>,
    corrections: CudaSlice<f32>,
    perfect_active: CudaSlice<u32>,
    histogram: CudaSlice<u32>,
}



impl NativeCudaDevice {
    pub fn new(init: &InitialChains, rng: CurandRng, seed: u64, ordinal: usize) -> Result<Self> {
        // Kernel-name suffix in jsv.cu.
        let suffix = rng.as_str();

        let ctx = open_context(ordinal)?;
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(cudarc::nvrtc::Ptx::from_src(JSV_PTX))
            .context("failed to load the JSV PTX module")?;

        let state_size = Self::query_state_size(&stream, &module, suffix)?;
        let states_bytes = state_size
            .checked_mul(init.num_chains)
            .context("cuRAND state buffer size overflowed")?;

        let mut rng_states = stream
            .alloc_zeros::<u8>(states_bytes)
            .context("failed to allocate cuRAND states")?;
        let init_fn = module
            .load_function(&format!("jsv_init_{suffix}"))
            .context("missing init kernel")?;
        let chains_u32 = init.num_chains as u32;
        let cfg = LaunchConfig {
            grid_dim: ((init.num_chains as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&init_fn);
        builder.arg(&mut rng_states).arg(&seed).arg(&chains_u32);
        unsafe { builder.launch(cfg) }.context("cuRAND init kernel failed")?;
        stream.synchronize().context("cuRAND init did not complete")?;

        // Stage the matchings in shared memory when a block's worth fits; the
        // per-chain arrays are hit several times per proposal and thousands of
        // times per launch, so keeping them off global memory pays for itself.
        let limit = ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        )? as usize;
        let want = shared_bytes(init.size);
        let staged = want <= limit;
        let (tag, staged_bytes) = if staged { ("", want) } else { ("_global", 0) };
        info!(
            "native CUDA backend on device {ordinal}: rng={} ({state_size}-byte states), \
             {} chains, matchings {} ({want} B/block, limit {limit} B)",
            rng.as_str(),
            init.num_chains,
            if staged { "in shared memory" } else { "in global memory" },
        );

        Ok(NativeCudaDevice {
            size: init.size,
            num_chains: init.num_chains,
            rng,
            warmup_fn: module.load_function(&format!("jsv_warmup{tag}_{suffix}"))?,
            occupancy_fn: module.load_function(&format!("jsv_occupancy{tag}_{suffix}"))?,
            ratio_fn: module.load_function(&format!("jsv_ratio{tag}_{suffix}"))?,
            shared_bytes: staged_bytes,
            row_match: stream.clone_htod(&init.row_match)?,
            col_match: stream.clone_htod(&init.col_match)?,
            holes_u: stream.clone_htod(&init.holes)?,
            holes_v: stream.clone_htod(&init.holes)?,
            adjacency: stream.clone_htod(&init.adjacency)?,
            active_counts: stream.clone_htod(&init.active_counts)?,
            rng_states,
            weights: stream.alloc_zeros::<f32>(init.size * init.size)?,
            exp_beta: stream.alloc_zeros::<f32>(5)?,
            next_weights: stream.alloc_zeros::<f32>(init.size * init.size)?,
            ratio_terms: stream.alloc_zeros::<f32>(init.size + 1)?,
            sums: stream.alloc_zeros::<f32>(init.num_chains)?,
            corrections: stream.alloc_zeros::<f32>(init.num_chains)?,
            perfect_active: stream.alloc_zeros::<u32>(init.num_chains)?,
            histogram: stream.alloc_zeros::<u32>(init.size * init.size + 1)?,
            module,
            stream,
        })
    }

    /// cuRAND state layouts are opaque and differ per generator, so ask the
    /// device rather than hardcoding sizes that vary across CUDA releases.
    fn query_state_size(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        suffix: &str,
    ) -> Result<usize> {
        let function = module
            .load_function(&format!("jsv_state_size_{suffix}"))
            .context("missing state-size kernel")?;
        let mut out = stream.alloc_zeros::<u32>(1)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&function);
        builder.arg(&mut out);
        unsafe { builder.launch(cfg) }.context("state-size kernel failed")?;
        let sizes = stream.clone_dtoh(&out)?;
        Ok(sizes[0] as usize)
    }

    fn launch_cfg(&self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((self.num_chains as u32).div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: self.shared_bytes as u32,
        }
    }

}

impl JsvDevice for NativeCudaDevice {
    fn name(&self) -> String {
        format!("native-cuda/{}", self.rng.as_str())
    }

    fn begin_step(&mut self, weights: &[f32], exp_beta: &[f32]) {
        let stream = self.stream.clone();
        stream
            .memcpy_htod(weights, &mut self.weights)
            .expect("failed to upload weights");
        stream
            .memcpy_htod(exp_beta, &mut self.exp_beta)
            .expect("failed to upload exp_beta");
    }

    fn warmup_pass(&mut self, iterations: usize) {
        let n = self.size as u32;
        let chains = self.num_chains as u32;
        let iterations = iterations as u64;
        let cfg = self.launch_cfg();
        // Cheap Arc/handle clones, so the fields below can be borrowed directly.
        let stream = self.stream.clone();
        let function = self.warmup_fn.clone();
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&mut self.row_match)
            .arg(&mut self.col_match)
            .arg(&self.weights)
            .arg(&self.adjacency)
            .arg(&self.exp_beta)
            .arg(&mut self.holes_u)
            .arg(&mut self.holes_v)
            .arg(&mut self.active_counts)
            .arg(&mut self.rng_states)
            .arg(&n)
            .arg(&chains)
            .arg(&iterations);
        unsafe { builder.launch(cfg) }.expect("warmup kernel failed");
        stream.synchronize().expect("warmup did not complete");
    }

    fn occupancy_pass(&mut self, samples: usize, interval: usize) -> Vec<u32> {
        let n = self.size as u32;
        let chains = self.num_chains as u32;
        let (samples, interval) = (samples as u64, interval as u64);
        let cfg = self.launch_cfg();
        let stream = self.stream.clone();
        let function = self.occupancy_fn.clone();
        stream
            .memset_zeros(&mut self.histogram)
            .expect("failed to clear histogram");
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&mut self.row_match)
            .arg(&mut self.col_match)
            .arg(&self.weights)
            .arg(&self.adjacency)
            .arg(&self.exp_beta)
            .arg(&mut self.holes_u)
            .arg(&mut self.holes_v)
            .arg(&mut self.active_counts)
            .arg(&mut self.rng_states)
            .arg(&mut self.histogram)
            .arg(&n)
            .arg(&chains)
            .arg(&samples)
            .arg(&interval);
        unsafe { builder.launch(cfg) }.expect("occupancy kernel failed");
        stream
            .clone_dtoh(&self.histogram)
            .expect("failed to read histogram")
    }

    fn ratio_pass(
        &mut self,
        next_weights: &[f32],
        ratio_terms: &[f32],
        samples: usize,
        interval: usize,
    ) -> (f64, usize, usize) {
        let n = self.size as u32;
        let chains = self.num_chains as u32;
        let (samples_u64, interval_u64) = (samples as u64, interval as u64);
        let cfg = self.launch_cfg();
        let stream = self.stream.clone();
        let function = self.ratio_fn.clone();

        stream
            .memcpy_htod(next_weights, &mut self.next_weights)
            .expect("failed to upload next_weights");
        stream
            .memcpy_htod(ratio_terms, &mut self.ratio_terms)
            .expect("failed to upload ratio_terms");
        // The kernel writes these outright, so they need no clearing.
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&mut self.row_match)
            .arg(&mut self.col_match)
            .arg(&self.weights)
            .arg(&self.next_weights)
            .arg(&self.adjacency)
            .arg(&self.exp_beta)
            .arg(&self.ratio_terms)
            .arg(&mut self.holes_u)
            .arg(&mut self.holes_v)
            .arg(&mut self.active_counts)
            .arg(&mut self.rng_states)
            .arg(&mut self.sums)
            .arg(&mut self.corrections)
            .arg(&mut self.perfect_active)
            .arg(&n)
            .arg(&chains)
            .arg(&samples_u64)
            .arg(&interval_u64);
        unsafe { builder.launch(cfg) }.expect("ratio kernel failed");

        let sums = stream.clone_dtoh(&self.sums).expect("read sums");
        let corrections = stream
            .clone_dtoh(&self.corrections)
            .expect("read corrections");
        let hits = stream
            .clone_dtoh(&self.perfect_active)
            .expect("read perfect_active");
        let total = sums
            .iter()
            .zip(&corrections)
            .map(|(&sum, &correction)| sum as f64 + correction as f64)
            .sum::<f64>();
        let hits = hits.iter().map(|&value| value as usize).sum::<usize>();
        (total, hits, self.num_chains * samples)
    }
}
