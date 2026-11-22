//! Sticky minimal navigation bar with URochester dimer mark, WebGPU status,
//! and adaptive theme toggle.

use dioxus::prelude::*;

use crate::state::{ResolvedTheme, ThemeMode, WorkerStatus};

#[component]
pub fn Navbar(
    theme_mode: ThemeMode,
    resolved_theme: ResolvedTheme,
    worker_status: WorkerStatus,
    on_cycle_theme: EventHandler<()>,
) -> Element {
    let (dot_class, gpu_label) = match &worker_status {
        WorkerStatus::Loading => ("gpu-dot pending", "Loading WebGPU…".to_owned()),
        WorkerStatus::Ready(caps) if caps.webgpu_available => {
            let label = match &caps.adapter_name {
                Some(name) if !name.is_empty() => format!("WebGPU · {name}"),
                _ => "WebGPU Ready".to_owned(),
            };
            ("gpu-dot ok", label)
        }
        WorkerStatus::Ready(_) => ("gpu-dot warn", "WebGPU Unavailable".to_owned()),
        WorkerStatus::Error(_) => ("gpu-dot warn", "Worker Offline".to_owned()),
    };

    let theme_icon = match (theme_mode, resolved_theme) {
        (ThemeMode::System, _) => "◐",
        (ThemeMode::Light, _) => "☀",
        (ThemeMode::Dark, _) => "☾",
    };

    rsx! {
        header { class: "navbar",
            div { class: "navbar-inner",
                a { class: "brand-link", href: "#top",
                    svg {
                        class: "brand-mark",
                        view_box: "0 0 24 24",
                        fill: "none",
                        rect { x: "1.5", y: "1.5", width: "21", height: "21", rx: "5", fill: "#001E5F" }
                        line { x1: "6.5", y1: "8", x2: "13.5", y2: "8", stroke: "#66A2FF", stroke_width: "2.6", stroke_linecap: "round" }
                        line { x1: "17.5", y1: "8", x2: "17.5", y2: "16", stroke: "#FFD82B", stroke_width: "2.6", stroke_linecap: "round" }
                        line { x1: "6.5", y1: "16", x2: "13.5", y2: "16", stroke: "#FFD82B", stroke_width: "2.6", stroke_linecap: "round" }
                        circle { cx: "6.5", cy: "8", r: "1.5", fill: "#FFFFFF" }
                        circle { cx: "17.5", cy: "16", r: "1.5", fill: "#FFFFFF" }
                    }
                    span { class: "brand-name", "permanent-rs" }
                    span { class: "brand-tag", "WGSL" }
                }

                div { class: "nav-actions",
                    div { class: "gpu-pill", title: "{gpu_label}",
                        span { class: "{dot_class}" }
                        span { class: "gpu-label-full", "{gpu_label}" }
                        span { class: "gpu-label-short",
                            if dot_class.contains("ok") { "WebGPU" } else if dot_class.contains("pending") { "GPU…" } else { "No GPU" }
                        }
                    }
                    button {
                        class: "icon-btn",
                        r#type: "button",
                        title: "Switch color theme (Auto / Light / Dark)",
                        onclick: move |_| on_cycle_theme.call(()),
                        span { "{theme_icon}" }
                        span { "{theme_mode.label()}" }
                    }
                    a {
                        class: "icon-btn",
                        href: "https://github.com/SchrodingerZhu/permanent-rs",
                        target: "_blank",
                        rel: "noreferrer",
                        title: "Source repository on GitHub",
                        span { class: "nav-github-label", "GitHub" }
                        span { "↗" }
                    }
                }
            }
        }
    }
}
