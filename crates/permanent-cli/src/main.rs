mod cli;
mod tui;

#[cfg(not(any(feature = "rayon", feature = "cuda", feature = "wgpu")))]
compile_error!("enable at least one engine: the `rayon`, `wgpu` or `cuda` feature");

#[cfg(not(target_family = "wasm"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> anyhow::Result<()> {
    cli::main()
}
