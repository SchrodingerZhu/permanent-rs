fn main() {
    // src/philox.rs of the parent crate marks its functions as device code
    // under this cfg.
    println!("cargo::rustc-cfg=cuda_kernel");
}
