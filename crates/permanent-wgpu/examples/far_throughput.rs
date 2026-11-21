//! Steps per second of the FAR kernel on a matrix, for comparing kernel
//! changes: `cargo run --release -p permanent-wgpu --example far_throughput
//! -- data/dimer-32x32.mtx [particles] [cycles]`.

#![cfg_attr(target_family = "wasm", allow(unused_imports))]

use std::time::Instant;

use permanent_core::engine::{Engine, Stream};
use permanent_core::problem::Problem;
use permanent_core::scaling::{Prepared, prepare};
use permanent_core::tables::Tables;
use permanent_wgpu::Wgpu;

/// A browser has no command line to run this from.
#[cfg(target_family = "wasm")]
fn main() {}

#[cfg(not(target_family = "wasm"))]
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("a matrix in Matrix Market format");
    let particles: usize = args.next().map_or(32768, |x| x.parse().unwrap());
    let cycles: u64 = args.next().map_or(512, |x| x.parse().unwrap());
    let working_mb: Option<usize> = args.next().map(|x| x.parse().unwrap());
    let calls: u64 = args.next().map_or(6, |x| x.parse().unwrap());
    let Prepared::Scaled(scaled) = prepare(&Problem::load(&path)?) else { anyhow::bail!("the permanent is zero") };
    let n = scaled.n;
    // at temperature zero the hole weights are n exactly
    let tables = Tables::<u64>::build(&scaled, 0.0, &vec![(n as f64).ln(); n], 0);
    let identity: Vec<u32> = (0..particles).flat_map(|_| 0..n as u32).collect();
    let mut engine = Wgpu::new(&scaled)?;
    engine.load(&identity, 0);
    if let Some(mb) = working_mb {
        engine.working_set = mb << 20;
    }
    println!(
        "{} on {}: {particles} particles, {cycles} cycles a call, working set {} MB",
        path,
        Engine::name(&engine),
        engine.working_set >> 20
    );
    for call in 0..calls {
        let clock = Instant::now();
        let round =
            Engine::far(&mut engine, &tables, cycles, u64::MAX >> 1, 64, Stream { seed: 1, phase: 1, level: call })?;
        let seconds = clock.elapsed().as_secs_f64();
        println!(
            "call {call}: {:.3e} steps in {seconds:.3} s, {:.3e} steps/s",
            round.steps as f64,
            round.steps as f64 / seconds
        );
    }
    Ok(())
}
