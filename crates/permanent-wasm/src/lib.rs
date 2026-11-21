//! Browser WebAssembly + WebGPU bridge for `permanent-rs`.
//!
//! Problem generators, exact references, and the standalone covering sampler
//! live in `permanent-demo`. This crate wraps them with the WebGPU-only
//! asynchronous telemetry driver ([`runner`]) and browser `#[wasm_bindgen]`
//! exports ([`wasm`] on `wasm32-unknown-unknown`).

pub mod runner;
pub mod types;

#[cfg(all(target_family = "wasm", target_os = "unknown"))]
pub mod wasm;

use types::{PresetInfo, WebRunConfig};

/// Return the curated demo catalog paired with default WebGPU run settings.
pub fn list_presets() -> Vec<PresetInfo> {
    permanent_demo::presets::catalog()
        .into_iter()
        .map(|preset| PresetInfo {
            id: preset.id,
            title: preset.title,
            subtitle: preset.subtitle,
            spec: preset.spec,
            recommended_config: WebRunConfig {
                epsilon: 0.1,
                delta: 0.05,
                seed: 1,
                compute_exact: preset.compute_exact,
                ..Default::default()
            },
        })
        .collect()
}

#[cfg(all(test, not(all(target_family = "wasm", target_os = "unknown"))))]
mod tests {
    use super::*;

    #[test]
    fn presets_match_demo_catalog_and_legacy_backend_field_is_ignored() {
        let presets = list_presets();
        assert_eq!(presets.len(), permanent_demo::presets::catalog().len());
        assert!(!presets.is_empty());

        let config: WebRunConfig = serde_json::from_str(r#"{"backend": "auto", "epsilon": 0.2}"#).unwrap();
        assert!((config.epsilon - 0.2).abs() < 1e-12);
    }
}
