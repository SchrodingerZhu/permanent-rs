mod cli;
mod tui;

#[cfg(not(any(feature = "rayon", feature = "cuda")))]
compile_error!("enable at least one engine: the `rayon` or the `cuda` feature");

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> anyhow::Result<()> {
    cli::main()
}
