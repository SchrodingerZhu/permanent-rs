use permanent::engine::Cpu;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> anyhow::Result<()> {
    permanent::cli::main(|scaled, _, _| Ok(Box::new(Cpu::new(scaled))))
}
