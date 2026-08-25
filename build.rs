//! Compiles the native CUDA kernels to PTX when the `native-cuda` feature is
//! on. Device-only (`nvcc --ptx`): no host CUDA code is compiled and nothing
//! links against libcudart, so the build never has to reconcile nvcc's host
//! toolchain with the one building the Rust crate. The PTX is loaded at run
//! time through the CUDA driver API.

use std::{env, path::PathBuf, process::Command};

const KERNEL: &str = "src/cuda/jsv.cu";
/// Lowest virtual architecture nvcc 13.x accepts. PTX is forward-compatible,
/// so this JITs onto everything up to and including Blackwell.
const DEFAULT_ARCH: &str = "compute_75";

fn main() {
    println!("cargo::rerun-if-changed={KERNEL}");
    println!("cargo::rerun-if-env-changed=CUDA_PATH");
    println!("cargo::rerun-if-env-changed=PERMANENT_CUDA_ARCH");

    // `cfg(feature = ...)` in a build script refers to the build script's own
    // dependencies, so the crate's features arrive as environment variables.
    if env::var_os("CARGO_FEATURE_NATIVE_CUDA").is_none() {
        return;
    }

    let cuda_path = env::var("CUDA_PATH").ok().map(PathBuf::from);
    let nvcc = locate_nvcc(cuda_path.as_deref());
    let arch = env::var("PERMANENT_CUDA_ARCH").unwrap_or_else(|_| DEFAULT_ARCH.to_string());
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("jsv.ptx");

    let mut command = Command::new(&nvcc);
    command
        .arg("--ptx")
        .arg("-O3")
        .arg(format!("-arch={arch}"))
        .arg(KERNEL)
        .arg("-o")
        .arg(&out);
    // nvcc resolves its default include directory relative to the real path of
    // its own binary, which misses the cudart/CRT/cuRAND headers whenever the
    // toolkit is assembled from split packages (as it is under nix).
    if let Some(path) = cuda_path.as_deref() {
        command.arg("-I").arg(path.join("include"));
    }

    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", nvcc.display()));
    assert!(
        status.success(),
        "{} failed to compile {KERNEL} for {arch}",
        nvcc.display()
    );

    println!("cargo::rustc-env=PERMANENT_JSV_PTX={}", out.display());
}

/// `CUDA_PATH/bin/nvcc` if it is set, otherwise whatever is on `PATH`.
fn locate_nvcc(cuda_path: Option<&std::path::Path>) -> PathBuf {
    if let Some(path) = cuda_path {
        let candidate = path.join("bin").join("nvcc");
        if candidate.exists() {
            return candidate;
        }
    }
    if Command::new("nvcc").arg("--version").output().is_ok() {
        return PathBuf::from("nvcc");
    }
    panic!(
        "the `native-cuda` feature needs nvcc to compile {KERNEL}, but none was \
         found. Set CUDA_PATH to a CUDA toolkit, or enter the flake's CUDA shell \
         with `nix develop .#cuda`. Build without `--features native-cuda` if you \
         do not need the native CUDA backend."
    );
}
