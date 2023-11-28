use std::num::NonZeroUsize;

use clap::{Parser, ValueEnum};
use filter::MetropolisFilter;
use tracing::{error, info, level_filters::LevelFilter};
use tracing_subscriber::EnvFilter;

use crate::{
    cooling_schedule::CoolingConfig,
    graph::Graph,
    markov_chain::{Config, MCState},
};

pub mod cooling_schedule;
pub mod cooling_state;
pub mod dinic;
pub mod filter;
pub mod graph;

pub mod markov_chain;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug)]
pub struct Cli {
    /// Path to the graph file.
    #[arg(short, long)]
    pub graph_path: std::path::PathBuf,
    /// Number of chains.
    #[arg(short, long, default_value_t = 2048)]
    pub num_of_chains: usize,
    /// Potential mixing time of initial runs.
    #[arg(short, long, default_value_t = 16384)]
    pub warmup_times: usize,
    /// Potential relaxation time of the chain (for weight estimation).
    #[arg(short = 'W', long, default_value_t = 16)]
    pub weight_sample_intervals: usize,
    /// Potential relaxation time of the chain (for estimator estimation).
    #[arg(short, long, default_value_t = 128)]
    pub estimator_sample_intervals: usize,
    /// Number of samples to from each chain for weight estimation.
    #[arg(short = 'q', long, default_value_t = 2048)]
    pub num_of_weight_estimations: usize,
    /// Number of samples to from each chain for estimator estimation.
    #[arg(short = 'p', long, default_value_t = 64)]
    pub num_of_estimator_estimations: usize,
    /// Number of threads to use (use all available threads if not specified).
    #[arg(short = 't', long)]
    pub num_of_threads: Option<usize>,
    /// Slow down factor of the additive increment.
    #[arg(long, default_value_t = NonZeroUsize::new(1).unwrap())]
    pub additive_slow_down: NonZeroUsize,
    /// Slow down factor of the multiplicative increment.
    #[arg(long, default_value_t = NonZeroUsize::new(1).unwrap())]
    pub mutiplicative_slow_down: NonZeroUsize,
    /// Metroplis filter to use.
    #[arg(short = 'f', long, default_value = "additive")]
    pub filter: Filter,
}

#[derive(Parser, Debug, ValueEnum, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    Additive,
    Multiplicative,
    Constant,
}

fn run_chain<F: MetropolisFilter + Send + Sync + 'static>(
    graph: Graph,
    config: Config,
    add_factor: NonZeroUsize,
    mul_factor: NonZeroUsize,
) {
    let size = graph.size;
    let mut state = MCState::<F>::new(graph, config);
    state.warmup();
    info!("Warmup finished");
    let cooling_cfg = CoolingConfig {
        n: NonZeroUsize::new(size).unwrap(),
        additive_ratio: add_factor,
        multiplicative_ratio: mul_factor,
    };
    let schedule = crate::cooling_schedule::CoolingSchedule::from(cooling_cfg);
    state.cooling_evolve(schedule, false);
    info!("final weight matrix:");
    for i in 0..size {
        for j in 0..size {
            // print state.global_state.weight.get(i, j)
            print!("{:.2} ", 1.0 / state.global_state.weight.get(i, j));
        }
        println!();
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .with_env_var("PERMANENT_LOG_LEVEL")
                .from_env_lossy(),
        )
        .init();
    let cli = Cli::parse();
    let thd_cnt = cli.num_of_threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1)
    });
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
    match cli.filter {
        Filter::Additive => run_chain::<filter::Additive>(
            graph,
            config,
            cli.additive_slow_down,
            cli.mutiplicative_slow_down,
        ),
        Filter::Multiplicative => run_chain::<filter::Multiplicative>(
            graph,
            config,
            cli.additive_slow_down,
            cli.mutiplicative_slow_down,
        ),
        Filter::Constant => run_chain::<filter::Constant>(
            graph,
            config,
            cli.additive_slow_down,
            cli.mutiplicative_slow_down,
        ),
    }
}
