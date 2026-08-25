use std::num::NonZeroUsize;

use clap::Parser;
use tracing::{error, info, level_filters::LevelFilter};
use tracing_subscriber::EnvFilter;

use crate::{
    cooling_schedule::CoolingConfig,
    cooling_state::State,
    gpu_markov_chain::{CubeclRng, CurandRng, GpuMCState},
    graph::Graph,
    markov_chain::{Config, MCState},
};

pub mod chain;
pub mod cooling_schedule;
pub mod cooling_state;
#[cfg(feature = "native-cuda")]
pub mod cuda_backend;
pub mod dinic;
pub mod exact;
pub mod gpu_markov_chain;
pub mod graph;

pub mod markov_chain;
pub mod tui;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Execution backend, naming both the device and the generator it draws from.
/// The two are chosen together because which generators exist depends on the
/// device: cuRAND's are reachable only from hand-written CUDA, and the CubeCL
/// kernels carry their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Multi-threaded CPU chains.
    Cpu,
    /// CubeCL kernels, 32-bit xorshift. Fast, but fails Crush; a speed baseline.
    CubeclXorshift,
    /// CubeCL kernels, Philox4x32-10. The portable default.
    CubeclPhilox,
    /// Native CUDA kernels, cuRAND `curandStatePhilox4_32_10_t`.
    CudaPhilox,
    /// Native CUDA kernels, cuRAND `curandStateXORWOW_t`.
    CudaXorwow,
    /// Native CUDA kernels, cuRAND `curandStateMRG32k3a_t`.
    CudaMrg32k3a,
}

impl Backend {
    const VALUES: [(&'static str, Backend); 6] = [
        ("cpu", Backend::Cpu),
        ("cubecl_xorshift", Backend::CubeclXorshift),
        ("cubecl_philox", Backend::CubeclPhilox),
        ("cuda_philox", Backend::CudaPhilox),
        ("cuda_xorwow", Backend::CudaXorwow),
        ("cuda_mrg32k3a", Backend::CudaMrg32k3a),
    ];

    pub fn as_str(self) -> &'static str {
        Self::VALUES
            .iter()
            .find(|(_, backend)| *backend == self)
            .map(|(name, _)| *name)
            .unwrap_or("cpu")
    }

    /// Whether this backend needs the `native-cuda` feature and a driver.
    pub fn is_native_cuda(self) -> bool {
        matches!(
            self,
            Backend::CudaPhilox | Backend::CudaXorwow | Backend::CudaMrg32k3a
        )
    }
}

impl std::str::FromStr for Backend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        // Accept hyphens too, so `cuda-philox` works as well as `cuda_philox`.
        let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
        Self::VALUES
            .iter()
            .find(|(name, _)| *name == normalized)
            .map(|(_, backend)| *backend)
            .ok_or_else(|| {
                let names = Self::VALUES
                    .iter()
                    .map(|(name, _)| *name)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("unknown backend `{value}`; expected one of: {names}")
            })
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Parser, Debug)]
pub struct Cli {
    /// Path to the graph file.
    #[arg(short, long)]
    pub graph_path: std::path::PathBuf,
    /// Markov-chain execution backend and generator: cpu, cubecl_xorshift,
    /// cubecl_philox, cuda_philox, cuda_xorwow, cuda_mrg32k3a. The cuda_*
    /// backends need a build with `--features native-cuda` and an NVIDIA driver.
    #[arg(long, default_value_t = Backend::Cpu)]
    pub backend: Backend,
    /// Seed for the device generators. Runs with the same seed and the same
    /// configuration reproduce each other.
    #[arg(long, default_value_t = 0x5eed_1e55_c0ff_ee01)]
    pub seed: u64,
    /// CUDA device ordinal for the cuda_* backends.
    #[arg(long, default_value_t = 0)]
    pub cuda_device: usize,
    /// Number of chains.
    #[arg(short, long, default_value_t = 2048)]
    pub num_of_chains: usize,
    /// Potential mixing time of initial runs.
    #[arg(short, long, default_value_t = 16384)]
    pub warmup_times: usize,
    /// Proposals between per-step samples. The JSV walker needs stirring
    /// that grows with the graph; values around n are a good starting point
    /// for large instances (the mixing-time guesswork lives here).
    #[arg(short = 'W', long, default_value_t = 16)]
    pub weight_sample_intervals: usize,
    /// Proposals between samples in the final perfect-fraction round.
    #[arg(short, long, default_value_t = 128)]
    pub estimator_sample_intervals: usize,
    /// Per-chain samples per cooling step; the first half bootstraps the
    /// hole-weight table, the second half feeds the ratio estimator.
    #[arg(short = 'q', long, default_value_t = 2048)]
    pub num_of_weight_estimations: usize,
    /// Minimum per-chain samples in the final perfect-fraction round.
    #[arg(short = 'p', long, default_value_t = 64)]
    pub num_of_estimator_estimations: usize,
    /// Number of threads to use (use all available threads if not specified).
    #[arg(short = 't', long)]
    pub num_of_threads: Option<usize>,
    /// Slow down factor of the additive increment.
    #[arg(long, default_value_t = NonZeroUsize::new(4).unwrap())]
    pub additive_slow_down: NonZeroUsize,
    /// Slow down factor of the multiplicative increment.
    #[arg(long, default_value_t = NonZeroUsize::new(4).unwrap())]
    pub multiplicative_slow_down: NonZeroUsize,
    /// Also compute the exact permanent via Ryser's formula (feasible for
    /// small graphs only) and report it alongside the estimate.
    #[arg(long, default_value_t = false)]
    pub exact: bool,
    /// Visualize the annealing process in a live terminal dashboard.
    #[arg(long, default_value_t = false)]
    pub tui: bool,
}

enum ChainState {
    Cpu(MCState),
    Gpu(Box<GpuMCState>),
}

/// Device knobs that only apply to the GPU backends.
#[derive(Debug, Clone, Copy)]
pub struct GpuOptions {
    pub seed: u64,
    pub cuda_device: usize,
}

impl ChainState {
    fn new(
        backend: Backend,
        graph: Graph,
        config: Config,
        options: GpuOptions,
    ) -> anyhow::Result<Self> {
        match backend {
            Backend::Cpu => Ok(ChainState::Cpu(MCState::new(graph, config))),
            Backend::CubeclXorshift => Ok(ChainState::Gpu(Box::new(GpuMCState::cubecl(
                graph,
                config,
                CubeclRng::Xorshift32,
                options.seed,
            )))),
            Backend::CubeclPhilox => Ok(ChainState::Gpu(Box::new(GpuMCState::cubecl(
                graph,
                config,
                CubeclRng::Philox,
                options.seed,
            )))),
            Backend::CudaPhilox => Self::new_native_cuda(graph, config, CurandRng::Philox, options),
            Backend::CudaXorwow => Self::new_native_cuda(graph, config, CurandRng::Xorwow, options),
            Backend::CudaMrg32k3a => {
                Self::new_native_cuda(graph, config, CurandRng::Mrg32k3a, options)
            }
        }
    }

    #[cfg(feature = "native-cuda")]
    fn new_native_cuda(
        graph: Graph,
        config: Config,
        rng: CurandRng,
        options: GpuOptions,
    ) -> anyhow::Result<Self> {
        use crate::cuda_backend::NativeCudaDevice;
        use crate::gpu_markov_chain::{InitialChains, JsvDevice};
        let state = GpuMCState::try_with_device(graph, config, |init: &InitialChains| {
            let device = NativeCudaDevice::new(init, rng, options.seed, options.cuda_device)?;
            Ok(Box::new(device) as Box<dyn JsvDevice>)
        })?;
        Ok(ChainState::Gpu(Box::new(state)))
    }

    #[cfg(not(feature = "native-cuda"))]
    fn new_native_cuda(
        _graph: Graph,
        _config: Config,
        _rng: CurandRng,
        _options: GpuOptions,
    ) -> anyhow::Result<Self> {
        anyhow::bail!(
            "this binary was built without the native CUDA backend. Rebuild with \
             `cargo build --release --features native-cuda` (which needs nvcc; \
             `nix develop .#cuda` provides one), or use `--backend cubecl_philox` \
             for the portable CubeCL path."
        )
    }

    fn warmup(&mut self) {
        match self {
            ChainState::Cpu(state) => state.warmup(),
            ChainState::Gpu(state) => state.warmup(),
        }
    }

    fn cooling_evolve_with<F>(
        &mut self,
        schedule: cooling_schedule::CoolingSchedule,
        observer: F,
    ) -> f64
    where
        F: FnMut(&markov_chain::StepStats, &State) -> bool,
    {
        match self {
            ChainState::Cpu(state) => state.cooling_evolve_with(schedule, observer),
            ChainState::Gpu(state) => state.cooling_evolve_with(schedule, observer),
        }
    }

    fn cooling_evolve(&mut self, schedule: cooling_schedule::CoolingSchedule) -> f64 {
        match self {
            ChainState::Cpu(state) => state.cooling_evolve(schedule),
            ChainState::Gpu(state) => state.cooling_evolve(schedule),
        }
    }

    fn global_state(&self) -> &State {
        match self {
            ChainState::Cpu(state) => &state.global_state,
            ChainState::Gpu(state) => &state.global_state,
        }
    }
}

fn make_schedule(
    size: usize,
    add_factor: NonZeroUsize,
    mul_factor: NonZeroUsize,
) -> cooling_schedule::CoolingSchedule {
    let cooling_cfg = CoolingConfig {
        n: NonZeroUsize::new(size).unwrap(),
        additive_ratio: add_factor,
        multiplicative_ratio: mul_factor,
    };
    cooling_schedule::CoolingSchedule::from(cooling_cfg)
}

fn run_chain(
    mut state: ChainState,
    size: usize,
    add_factor: NonZeroUsize,
    mul_factor: NonZeroUsize,
) -> f64 {
    state.warmup();
    info!("Warmup finished");
    let schedule = make_schedule(size, add_factor, mul_factor);
    let estimator = state.cooling_evolve(schedule);
    info!("final hole-class abundances (1/w):");
    for i in 0..size {
        for j in 0..size {
            print!("{:.2} ", 1.0 / state.global_state().weight.get(i, j));
        }
        println!();
    }
    estimator
}

fn run_chain_tui(
    mut state: ChainState,
    size: usize,
    adjacency: Vec<bool>,
    add_factor: NonZeroUsize,
    mul_factor: NonZeroUsize,
    exact: Option<f64>,
) -> Option<f64> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(tui::TuiEvent::Init { n: size, adjacency });
        let _ = tx.send(tui::TuiEvent::WarmupStarted);
        state.warmup();
        let schedule = make_schedule(size, add_factor, mul_factor);
        let estimator = state.cooling_evolve_with(schedule, |step, global| {
            let mut marginals = vec![0.0; size * size];
            for (index, value) in marginals.iter_mut().enumerate() {
                *value = 1.0 / global.weight_of_edge(index / size, index % size);
            }
            tx.send(tui::TuiEvent::Step(tui::StepUpdate {
                step: step.step,
                total_steps: step.total_steps,
                beta: step.beta,
                ratio: step.ratio,
                estimator: step.estimator,
                acceptance: step.accepted_samples / step.attempted_samples.max(1) as f64,
                marginals,
            }))
            .is_ok()
        });
        let _ = tx.send(tui::TuiEvent::Done { estimator });
    });
    tui::run(rx, exact).expect("terminal error")
}

fn main() {
    let cli = Cli::parse();
    if !cli.tui {
        // the log stream would fight the dashboard for the terminal
        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::builder()
                    .with_default_directive(LevelFilter::INFO.into())
                    .with_env_var("PERMANENT_LOG_LEVEL")
                    .from_env_lossy(),
            )
            .init();
    }
    let thd_cnt = cli.num_of_threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1)
    });
    info!("Using {} threads", thd_cnt);
    info!("Using {} Markov-chain backend", cli.backend);
    rayon::ThreadPoolBuilder::new()
        .num_threads(thd_cnt)
        .build_global()
        .unwrap();
    let graph = Graph::load(cli.graph_path).unwrap();
    info!("Graph loaded: {:?}", graph);
    if graph.find_match().size() != graph.size {
        error!("Graph does not have a perfect matching");
        return;
    }
    let config = Config {
        num_of_chains: cli.num_of_chains,
        warmup_times: cli.warmup_times,
        weight_sample_intervals: cli.weight_sample_intervals,
        estimator_sample_intervals: cli.estimator_sample_intervals,
        num_of_weight_estimations: cli.num_of_weight_estimations,
        num_of_estimator_estimations: cli.num_of_estimator_estimations,
    };
    info!(
        "additive increment is slow down by {}",
        cli.additive_slow_down
    );
    info!(
        "multiplicative increment is slow down by {}",
        cli.multiplicative_slow_down
    );
    info!("{:#?}", config);
    let exact = cli.exact.then(|| exact::permanent(&graph));
    let exact_f64 = exact.as_ref().map(exact::to_f64);
    let size = graph.size;
    let adjacency = {
        let mut adjacency = vec![false; size * size];
        for (u, edges) in graph.edges.iter().enumerate() {
            for v in edges.iter().copied() {
                adjacency[u * size + v] = true;
            }
        }
        adjacency
    };
    // Built before the dashboard takes the terminal, so a backend that cannot
    // start reports plainly instead of from behind the TUI.
    let options = GpuOptions {
        seed: cli.seed,
        cuda_device: cli.cuda_device,
    };
    let state = match ChainState::new(cli.backend, graph, config, options) {
        Ok(state) => state,
        Err(problem) => {
            if cli.tui {
                eprintln!("error: {problem:#}");
            } else {
                error!("{problem:#}");
            }
            std::process::exit(1);
        }
    };
    if cli.tui {
        let estimator = run_chain_tui(
            state,
            size,
            adjacency,
            cli.additive_slow_down,
            cli.multiplicative_slow_down,
            exact_f64,
        );
        // the subscriber is not installed in TUI mode; print plainly
        match estimator {
            Some(estimator) => {
                println!("estimated permanent: {estimator:.6e}");
                if let Some(exact) = exact.as_ref() {
                    let exact_f64 = exact::to_f64(exact);
                    println!(
                        "exact permanent (Ryser): {exact}, relative error: {:.2}%",
                        (estimator - exact_f64).abs() / exact_f64 * 100.0
                    );
                }
            }
            None => println!("annealing interrupted before completion"),
        }
        return;
    }
    let estimator = run_chain(
        state,
        size,
        cli.additive_slow_down,
        cli.multiplicative_slow_down,
    );
    info!("estimated permanent: {estimator:.6e}");
    if let Some(exact) = exact.as_ref() {
        let exact_f64 = exact::to_f64(exact);
        info!(
            "exact permanent (Ryser): {exact}, relative error: {:.2}%",
            (estimator - exact_f64).abs() / exact_f64 * 100.0
        );
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;
    use std::str::FromStr;

    fn parse_backend(value: &str) -> Backend {
        Cli::try_parse_from(["permanent", "--graph-path", "g.json", "--backend", value])
            .unwrap()
            .backend
    }

    #[test]
    fn backend_defaults_to_cpu() {
        let default = Cli::try_parse_from(["permanent", "--graph-path", "g.json"]).unwrap();
        assert_eq!(default.backend, Backend::Cpu);
    }

    #[test]
    fn every_backend_round_trips_through_its_name() {
        for (name, expected) in Backend::VALUES {
            assert_eq!(parse_backend(name), expected, "parsing {name}");
            assert_eq!(expected.as_str(), name, "displaying {name}");
        }
    }

    #[test]
    fn backend_names_accept_hyphens_and_case() {
        assert_eq!(parse_backend("cuda-mrg32k3a"), Backend::CudaMrg32k3a);
        assert_eq!(parse_backend("CUBECL_PHILOX"), Backend::CubeclPhilox);
    }

    #[test]
    fn unknown_backend_lists_the_valid_names() {
        let error = Backend::from_str("gpu").unwrap_err();
        assert!(error.contains("unknown backend `gpu`"), "{error}");
        for (name, _) in Backend::VALUES {
            assert!(error.contains(name), "{error} should mention {name}");
        }
    }

    #[test]
    fn cuda_backends_are_flagged_as_native() {
        for (name, backend) in Backend::VALUES {
            assert_eq!(
                backend.is_native_cuda(),
                name.starts_with("cuda_"),
                "{name}"
            );
        }
    }
}
