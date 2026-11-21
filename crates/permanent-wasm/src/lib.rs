//! Browser WebAssembly + WebGPU bridge for `permanent-rs`.
//!
//! Pure-Rust demo generators, standalone covering exploration, and the
//! asynchronous telemetry driver live in [`generators`], [`sample`], and
//! [`runner`], so they build and test cleanly on the host. Browser
//! `#[wasm_bindgen]` entry points live in [`wasm`] on `wasm32-unknown-unknown`.

pub mod generators;
pub mod runner;
pub mod sample;
pub mod types;

#[cfg(all(target_family = "wasm", target_os = "unknown"))]
pub mod wasm;
