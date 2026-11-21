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
    use crate::types::{Phase1Summary, PlanSummary, WebEvent};

    #[test]
    fn presets_match_demo_catalog_and_legacy_backend_field_is_ignored() {
        let presets = list_presets();
        assert_eq!(presets.len(), permanent_demo::presets::catalog().len());
        assert!(!presets.is_empty());

        let config: WebRunConfig = serde_json::from_str(r#"{"backend": "auto", "epsilon": 0.2}"#).unwrap();
        assert!((config.epsilon - 0.2).abs() < 1e-12);
    }

    #[test]
    fn telemetry_events_round_trip_through_serde() {
        let events = [
            WebEvent::Phase1(Phase1Summary {
                temperatures: 4,
                seconds: 0.5,
                learn_particles: 128,
                learn_iterations: 8,
                far_steps: 256,
                mutate_events: 0,
                mutations: 0,
                mutate_time: 0.0,
                learn_mutation_time: 0.0,
                learn_capped_chains: 0,
                mutate_over_budget: 0,
                worst_first_correction_factor: 1.05,
                check_min: 0.98,
                check_max: 1.02,
            }),
            WebEvent::Plan(PlanSummary {
                segments: vec![[0, 2], [2, 4]],
                ln_scale: 12.3,
                particles: 32,
                islands: 16,
                relvar_sum: 0.4,
                mixing_factor: 1.1,
                island_relvar: 0.8,
                stage_time: 0.25,
                predicted_events: 1000.0,
                sigma_target: 0.05,
                z: 1.96,
            }),
            WebEvent::Round {
                round: 0,
                islands: 16,
                ln_permanent: 2.2,
                std_error: 0.04,
                mutate_events: 500,
                sampler_events: 100,
                seconds: 0.3,
            },
        ];
        for event in events {
            let encoded = serde_json::to_string(&event).unwrap();
            let decoded: WebEvent = serde_json::from_str(&encoded).unwrap();
            assert_eq!(serde_json::to_value(&event).unwrap(), serde_json::to_value(&decoded).unwrap());
        }
    }
}
