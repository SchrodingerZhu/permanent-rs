use std::num::NonZeroUsize;

use clap::Parser;
use tracing::{error, info, level_filters::LevelFilter};
use tracing_subscriber::EnvFilter;

use crate::{
    cooling_schedule::CoolingConfig,
    graph::Graph,
    markov_chain::{Config, MCState},
};

pub mod chain;
pub mod cooling_schedule;
pub mod cooling_state;
pub mod dinic;
pub mod exact;
pub mod graph;

pub mod markov_chain;
pub mod tui;

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
    graph: Graph,
    config: Config,
    add_factor: NonZeroUsize,
    mul_factor: NonZeroUsize,
) -> f64 {
    let size = graph.size;
    let mut state = MCState::new(graph, config);
    state.warmup();
    info!("Warmup finished");
    let schedule = make_schedule(size, add_factor, mul_factor);
    let estimator = state.cooling_evolve(schedule);
    info!("final weight matrix:");
    for i in 0..size {
        for j in 0..size {
            // print state.global_state.weight.get(i, j)
            print!("{:.2} ", 1.0 / state.global_state.weight.get(i, j));
        }
        println!();
    }
    estimator
}

fn run_chain_tui(
    graph: Graph,
    config: Config,
    add_factor: NonZeroUsize,
    mul_factor: NonZeroUsize,
    exact: Option<f64>,
) -> Option<f64> {
    let size = graph.size;
    let (tx, rx) = std::sync::mpsc::channel();
    let adjacency = {
        let mut adjacency = vec![false; size * size];
        for (u, edges) in graph.edges.iter().enumerate() {
            for v in edges.iter().copied() {
                adjacency[u * size + v] = true;
            }
        }
        adjacency
    };
    std::thread::spawn(move || {
        let _ = tx.send(tui::TuiEvent::Init { n: size, adjacency });
        let mut state = MCState::new(graph, config);
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
    if cli.tui {
        let estimator = run_chain_tui(
            graph,
            config,
            cli.additive_slow_down,
            cli.multiplicative_slow_down,
            exact,
        );
        // the subscriber is not installed in TUI mode; print plainly
        match estimator {
            Some(estimator) => {
                println!("estimated permanent: {estimator:.6e}");
                if let Some(exact) = exact {
                    println!(
                        "exact permanent (Ryser): {exact:.6e}, relative error: {:.2}%",
                        (estimator - exact).abs() / exact * 100.0
                    );
                }
            }
            None => println!("annealing interrupted before completion"),
        }
        return;
    }
    let estimator = run_chain(
        graph,
        config,
        cli.additive_slow_down,
        cli.multiplicative_slow_down,
    );
    info!("estimated permanent: {estimator:.6e}");
    if let Some(exact) = exact {
        info!(
            "exact permanent (Ryser): {exact:.6e}, relative error: {:.2}%",
            (estimator - exact).abs() / exact * 100.0
        );
    }
}
