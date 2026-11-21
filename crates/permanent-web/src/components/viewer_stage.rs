//! Monograph-framed left visualization stage dispatching to 3D Three.js,
//! 2D lattice, disjoint cycles, or weighted matrix heatmap renderers.

use dioxus::prelude::*;
use permanent_demo::types::{GeneratedProblem, ProblemGeometry, SampleSnapshot};

use crate::components::viewer_2d::Viewer2d;
use crate::components::viewer_3d::Viewer3d;
use crate::components::viewer_cycles::ViewerCycles;
use crate::components::viewer_matrix::ViewerMatrix;
use crate::state::ResolvedTheme;

#[component]
pub fn ViewerStage(
    problem: Option<GeneratedProblem>,
    displayed_sample: Option<SampleSnapshot>,
    showing_live_sample: bool,
    sampling_busy: bool,
    build_error: Option<String>,
    resolved_theme: ResolvedTheme,
    on_random_covering: EventHandler<()>,
    on_heaviest_matching: EventHandler<()>,
) -> Element {
    if let Some(err) = build_error {
        return rsx! {
            div { class: "panel-frame stage-column",
                div { class: "stage-header",
                    div { class: "stage-meta",
                        h3 { "Invalid Problem Specification" }
                    }
                }
                div { class: "panel-scroll",
                    div { class: "notice-banner err", "{err}" }
                }
            }
        };
    }

    let Some(info) = problem else {
        return rsx! {
            div { class: "panel-frame stage-column",
                div { class: "stage-header",
                    div { class: "stage-meta",
                        h3 { "Preparing model…" }
                    }
                }
                div { class: "viewport-body",
                    div { class: "empty-telemetry", style: "margin: 4rem auto; max-width: 22rem;",
                        "Building lattice geometry and Hungarian scaling…"
                    }
                }
            }
        };
    };

    let active_matching: Option<Vec<u32>> =
        displayed_sample.as_ref().map(|s| s.matching.clone()).or_else(|| info.heaviest_matching.clone());
    let active_valid = displayed_sample.as_ref().map(|s| s.valid).unwrap_or(true);
    let can_sample = info.has_perfect_matching && !sampling_busy;

    rsx! {
        div { class: "panel-frame stage-column",
            div { class: "stage-header",
                div { class: "stage-meta",
                    h3 { "{info.title}" }
                    div { class: "stage-chips",
                        span { class: "chip mono", "n = {info.n}" }
                        span { class: "chip mono", "{info.nnz} nonzeros" }
                        span { class: "chip", if info.is_zero_one { "0/1" } else { "weighted" } }
                        if !info.has_perfect_matching {
                            span { class: "chip warn", "permanent = 0" }
                        } else if let Some(ref sci) = info.exact_scientific {
                            span { class: "chip exact mono", "exact: {sci}" }
                        }
                        if let Some(ref s) = displayed_sample && showing_live_sample {
                            span { class: if s.valid { "chip live mono" } else { "chip warn mono" },
                                "live β = {s.beta:.2}"
                                if !s.valid { " · raised edge" }
                            }
                        }
                    }
                }

                div { class: "stage-actions",
                    button {
                        class: "btn-secondary",
                        r#type: "button",
                        disabled: !can_sample,
                        onclick: move |_| on_random_covering.call(()),
                        if sampling_busy { "Sampling…" } else { "Random Covering" }
                    }
                    button {
                        class: "btn-secondary",
                        r#type: "button",
                        disabled: info.heaviest_matching.is_none(),
                        onclick: move |_| on_heaviest_matching.call(()),
                        "Heaviest"
                    }
                }
            }

            match &info.geometry {
                ProblemGeometry::Lattice3d { .. } => rsx! {
                    Viewer3d {
                        geometry: info.geometry.clone(),
                        matching: active_matching,
                        valid: active_valid,
                        resolved_theme,
                    }
                },
                ProblemGeometry::Lattice2d { .. } => rsx! {
                    Viewer2d {
                        geometry: info.geometry.clone(),
                        matching: active_matching,
                        is_zero_one: info.is_zero_one,
                    }
                },
                ProblemGeometry::Cycles { .. } => rsx! {
                    ViewerCycles {
                        geometry: info.geometry.clone(),
                        matching: active_matching,
                    }
                },
                ProblemGeometry::Matrix => rsx! {
                    ViewerMatrix {
                        n: info.n,
                        triplets: info.triplets.clone(),
                        matching: active_matching,
                    }
                },
            }

            div { class: "stage-footer",
                span { "{info.description}" }
                if let Some(ref src) = info.exact_source {
                    span { class: "mono", "Reference: {src}" }
                }
            }
        }
    }
}
