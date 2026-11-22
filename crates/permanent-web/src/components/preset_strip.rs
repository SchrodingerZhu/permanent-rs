//! Visual diagram-driven demo selector strip populated dynamically from
//! `permanent_wasm::list_presets()`.

use dioxus::prelude::*;
use permanent_demo::types::ProblemSpec;
use permanent_wasm::types::PresetInfo;

fn short_card_labels(preset: &PresetInfo) -> (&'static str, &'static str) {
    match preset.id.as_str() {
        "cubic-open-4x4x4" => ("3D Open Cube", "4×4×4"),
        "cubic-torus-6x6x6" => ("3D Torus", "6×6×6"),
        "chessboard-8x8" => ("Chessboard", "8×8"),
        "square-weighted-16x16" => ("Square Dimers", "16×16"),
        "four-cycles-64" => ("Four Cycles", "64×64"),
        "weighted-16" => ("Weighted Matrix", "16×16"),
        "random-sparse-16" => ("Random Sparse", "16×16"),
        _ => ("Demo", ""),
    }
}

#[component]
fn PresetDiagram(id: String, spec: ProblemSpec) -> Element {
    match (id.as_str(), &spec) {
        // 3D Open Cube: crisp isometric lattice with X/Y/Z blue-azure-gold dimers and ●/○ sites
        (_, ProblemSpec::Dimer3d { periodic: false, .. }) => rsx! {
            svg { class: "preset-diagram", view_box: "0 0 88 52", fill: "none",
                // Back skeleton edges
                path {
                    d: "M44 6 L66 15 L66 37 L44 46 L22 37 L22 15 Z M44 6 L44 28 M44 28 L22 37 M44 28 L66 37 M22 15 L44 24 L66 15 M44 24 L44 46",
                    stroke: "var(--lattice-wire)", stroke_width: "1.1",
                }
                // Interior midline wireframe
                path {
                    d: "M33 10.5 L55 19.5 L55 41.5 M55 10.5 L33 19.5 L33 41.5 M22 26 L44 35 L66 26",
                    stroke: "var(--lattice-wire)", stroke_width: "0.9", stroke_opacity: "0.8",
                }
                // Active 3-axis dimers (Navy X, Azure Y, Gold Z)
                line { x1: "22", y1: "15", x2: "22", y2: "26", stroke: "var(--dimer-z)", stroke_width: "3.2", stroke_linecap: "round" }
                line { x1: "44", y1: "24", x2: "44", y2: "35", stroke: "var(--dimer-z)", stroke_width: "3.2", stroke_linecap: "round" }
                line { x1: "66", y1: "26", x2: "66", y2: "37", stroke: "var(--dimer-z)", stroke_width: "3.2", stroke_linecap: "round" }
                line { x1: "33", y1: "10.5", x2: "44", y2: "6", stroke: "var(--dimer-y)", stroke_width: "3.2", stroke_linecap: "round" }
                line { x1: "55", y1: "19.5", x2: "66", y2: "15", stroke: "var(--dimer-y)", stroke_width: "3.2", stroke_linecap: "round" }
                line { x1: "22", y1: "37", x2: "33", y2: "41.5", stroke: "var(--dimer-x)", stroke_width: "3.2", stroke_linecap: "round" }
                line { x1: "44", y1: "46", x2: "55", y2: "41.5", stroke: "var(--dimer-y)", stroke_width: "3.2", stroke_linecap: "round" }
                line { x1: "33", y1: "19.5", x2: "44", y2: "24", stroke: "var(--dimer-x)", stroke_width: "3.2", stroke_linecap: "round" }
                // Key bipartite vertices (● / ○)
                circle { cx: "44", cy: "6", r: "2.1", fill: "var(--node-row-fill)" }
                circle { cx: "22", cy: "15", r: "2.1", fill: "var(--node-col-fill)", stroke: "var(--node-col-stroke)", stroke_width: "1.1" }
                circle { cx: "66", cy: "15", r: "2.1", fill: "var(--node-col-fill)", stroke: "var(--node-col-stroke)", stroke_width: "1.1" }
                circle { cx: "44", cy: "24", r: "2.1", fill: "var(--node-row-fill)" }
                circle { cx: "44", cy: "46", r: "2.1", fill: "var(--node-col-fill)", stroke: "var(--node-col-stroke)", stroke_width: "1.1" }
                circle { cx: "22", cy: "37", r: "2.1", fill: "var(--node-row-fill)" }
                circle { cx: "66", cy: "37", r: "2.1", fill: "var(--node-row-fill)" }
            }
        },

        // 3D Periodic Torus: isometric lattice with crisp outward wrap-around stubs
        (_, ProblemSpec::Dimer3d { periodic: true, .. }) => rsx! {
            svg { class: "preset-diagram", view_box: "0 0 88 52", fill: "none",
                // Periodic boundary stubs (dashed outward)
                line { x1: "25", y1: "17", x2: "15", y2: "13", stroke: "var(--dimer-x)", stroke_width: "1.6", stroke_dasharray: "2.5 2" }
                line { x1: "63", y1: "35", x2: "73", y2: "39", stroke: "var(--dimer-x)", stroke_width: "1.6", stroke_dasharray: "2.5 2" }
                line { x1: "63", y1: "17", x2: "73", y2: "13", stroke: "var(--dimer-y)", stroke_width: "1.6", stroke_dasharray: "2.5 2" }
                line { x1: "25", y1: "35", x2: "15", y2: "39", stroke: "var(--dimer-y)", stroke_width: "1.6", stroke_dasharray: "2.5 2" }
                line { x1: "44", y1: "9", x2: "44", y2: "2", stroke: "var(--dimer-z)", stroke_width: "1.6", stroke_dasharray: "2.5 2" }
                line { x1: "44", y1: "43", x2: "44", y2: "50", stroke: "var(--dimer-z)", stroke_width: "1.6", stroke_dasharray: "2.5 2" }
                // Cube skeleton
                path {
                    d: "M44 9 L63 17 L63 35 L44 43 L25 35 L25 17 Z M25 17 L44 25 L63 17 M44 25 L44 43",
                    stroke: "var(--lattice-wire)", stroke_width: "1.1",
                }
                // Active dimers
                line { x1: "25", y1: "17", x2: "44", y2: "25", stroke: "var(--dimer-x)", stroke_width: "3.1", stroke_linecap: "round" }
                line { x1: "44", y1: "9", x2: "63", y2: "17", stroke: "var(--dimer-x)", stroke_width: "3.1", stroke_linecap: "round" }
                line { x1: "63", y1: "25", x2: "63", y2: "35", stroke: "var(--dimer-z)", stroke_width: "3.1", stroke_linecap: "round" }
                line { x1: "25", y1: "25", x2: "25", y2: "35", stroke: "var(--dimer-z)", stroke_width: "3.1", stroke_linecap: "round" }
                line { x1: "44", y1: "34", x2: "44", y2: "43", stroke: "var(--dimer-z)", stroke_width: "3.1", stroke_linecap: "round" }
                // Nodes
                circle { cx: "44", cy: "9", r: "2", fill: "var(--node-row-fill)" }
                circle { cx: "25", cy: "17", r: "2", fill: "var(--node-col-fill)", stroke: "var(--node-col-stroke)", stroke_width: "1.1" }
                circle { cx: "63", cy: "17", r: "2", fill: "var(--node-col-fill)", stroke: "var(--node-col-stroke)", stroke_width: "1.1" }
                circle { cx: "44", cy: "25", r: "2", fill: "var(--node-row-fill)" }
                circle { cx: "44", cy: "43", r: "2", fill: "var(--node-col-fill)", stroke: "var(--node-col-stroke)", stroke_width: "1.1" }
            }
        },

        // Chessboard Dominoes: clean 4×4 checkered grid with horizontal blue & vertical gold dominoes
        ("chessboard-8x8", _) => rsx! {
            svg { class: "preset-diagram", view_box: "0 0 88 52", fill: "none",
                // Outer board (44×44 centered at x=22..66, y=4..48)
                rect { x: "22", y: "4", width: "44", height: "44", rx: "2", fill: "var(--bg-subtle)", stroke: "var(--border-strong)", stroke_width: "1.1" }
                // Horizontal Navy dominoes
                rect { x: "24", y: "6", width: "18.5", height: "7.5", rx: "2.5", fill: "var(--accent-blue-tint)", stroke: "var(--dimer-x)", stroke_width: "1.5" }
                rect { x: "24", y: "17", width: "18.5", height: "7.5", rx: "2.5", fill: "var(--accent-blue-tint)", stroke: "var(--dimer-x)", stroke_width: "1.5" }
                rect { x: "45.5", y: "28", width: "18.5", height: "7.5", rx: "2.5", fill: "var(--accent-blue-tint)", stroke: "var(--dimer-x)", stroke_width: "1.5" }
                rect { x: "45.5", y: "39", width: "18.5", height: "7.5", rx: "2.5", fill: "var(--accent-blue-tint)", stroke: "var(--dimer-x)", stroke_width: "1.5" }
                // Vertical Gold dominoes
                rect { x: "45.5", y: "6", width: "7.5", height: "18.5", rx: "2.5", fill: "var(--accent-gold-tint)", stroke: "var(--dimer-z)", stroke_width: "1.5" }
                rect { x: "56.5", y: "6", width: "7.5", height: "18.5", rx: "2.5", fill: "var(--accent-gold-tint)", stroke: "var(--dimer-z)", stroke_width: "1.5" }
                rect { x: "24", y: "28", width: "7.5", height: "18.5", rx: "2.5", fill: "var(--accent-gold-tint)", stroke: "var(--dimer-z)", stroke_width: "1.5" }
                rect { x: "35", y: "28", width: "7.5", height: "18.5", rx: "2.5", fill: "var(--accent-gold-tint)", stroke: "var(--dimer-z)", stroke_width: "1.5" }
            }
        },

        // Weighted 2D Square Dimers: bipartite lattice with solid horizontal & dashed vertical bonds + dimers
        (_, ProblemSpec::Dimer2d { .. }) => rsx! {
            svg { class: "preset-diagram", view_box: "0 0 88 52", fill: "none",
                // Horizontal bonds (solid)
                for row in 0..3 {
                    line {
                        x1: "16", y1: "{10 + row * 16}", x2: "72", y2: "{10 + row * 16}",
                        stroke: "var(--lattice-wire)", stroke_width: "1.1",
                    }
                }
                // Vertical bonds (dashed, weight b = 0.5)
                for col in 0..4 {
                    line {
                        x1: "{16.0 + col as f64 * 18.6:.1}", y1: "10", x2: "{16.0 + col as f64 * 18.6:.1}", y2: "42",
                        stroke: "var(--lattice-wire)", stroke_width: "1.1", stroke_dasharray: "3 2.5",
                    }
                }
                // Active weighted dimers
                line { x1: "16", y1: "10", x2: "34.6", y2: "10", stroke: "var(--dimer-x)", stroke_width: "3.6", stroke_linecap: "round" }
                line { x1: "53.3", y1: "10", x2: "72", y2: "10", stroke: "var(--dimer-x)", stroke_width: "3.6", stroke_linecap: "round" }
                line { x1: "34.6", y1: "26", x2: "53.3", y2: "26", stroke: "var(--dimer-x)", stroke_width: "3.6", stroke_linecap: "round" }
                line { x1: "16", y1: "26", x2: "16", y2: "42", stroke: "var(--dimer-z)", stroke_width: "3.6", stroke_linecap: "round" }
                line { x1: "72", y1: "26", x2: "72", y2: "42", stroke: "var(--dimer-z)", stroke_width: "3.6", stroke_linecap: "round" }
                line { x1: "34.6", y1: "42", x2: "53.3", y2: "42", stroke: "var(--dimer-x)", stroke_width: "3.6", stroke_linecap: "round" }
                // Bipartite nodes
                for row in 0..3 {
                    for col in 0..4 {
                        {
                            let cx = 16.0 + col as f64 * 18.66;
                            let cy = 10.0 + row as f64 * 16.0;
                            let even = (row + col) % 2 == 0;
                            rsx! {
                                circle {
                                    cx: "{cx:.1}", cy: "{cy:.1}", r: "2.3",
                                    fill: if even { "var(--node-row-fill)" } else { "var(--node-col-fill)" },
                                    stroke: if even { "none" } else { "var(--node-col-stroke)" },
                                    stroke_width: "1.2",
                                }
                            }
                        }
                    }
                }
            }
        },

        // Four Separate Cycles: 4 rings showing choice 1 (blue) vs choice 2 (gold)
        (_, ProblemSpec::DisjointCycles { .. }) => rsx! {
            svg { class: "preset-diagram", view_box: "0 0 88 52", fill: "none",
                for (i, &(cx, cy, alt)) in [(26.0_f64, 15.0_f64, false), (62.0, 15.0, true), (26.0, 38.0, true), (62.0, 38.0, false)].iter().enumerate() {
                    {
                        let stroke = if alt { "var(--dimer-z)" } else { "var(--dimer-x)" };
                        rsx! {
                            g { key: "mc-{i}",
                                circle { cx: "{cx}", cy: "{cy}", r: "9.5", stroke: "var(--lattice-wire)", stroke_width: "1.1" }
                                // Alternating matched arcs around the cycle
                                circle {
                                    cx: "{cx}", cy: "{cy}", r: "9.5",
                                    stroke: "{stroke}", stroke_width: "2.7",
                                    stroke_dasharray: "5.2 4.7",
                                    stroke_dashoffset: if alt { "4.8" } else { "0" },
                                    stroke_linecap: "round",
                                }
                                circle { cx: "{cx}", cy: "{cy - 9.5}", r: "1.6", fill: "var(--node-row-fill)" }
                                circle { cx: "{cx + 9.5}", cy: "{cy}", r: "1.6", fill: "var(--node-col-fill)", stroke: "var(--node-col-stroke)", stroke_width: "0.9" }
                                circle { cx: "{cx}", cy: "{cy + 9.5}", r: "1.6", fill: "var(--node-row-fill)" }
                                circle { cx: "{cx - 9.5}", cy: "{cy}", r: "1.6", fill: "var(--node-col-fill)", stroke: "var(--node-col-stroke)", stroke_width: "0.9" }
                            }
                        }
                    }
                }
            }
        },

        // Weighted Matrix: 4×4 log-shaded blue heatmap with gold matching squares
        (_, ProblemSpec::BundledPreset { .. }) => rsx! {
            svg { class: "preset-diagram", view_box: "0 0 88 52", fill: "none",
                rect { x: "22", y: "4", width: "44", height: "44", rx: "2", fill: "var(--bg-subtle)", stroke: "var(--border-strong)", stroke_width: "1.1" }
                for (r, c, op) in [
                    (0, 0, 0.92), (0, 1, 0.35), (0, 3, 0.55),
                    (1, 0, 0.25), (1, 1, 0.72), (1, 2, 0.85),
                    (2, 1, 0.45), (2, 2, 0.65), (2, 3, 0.95),
                    (3, 0, 0.68), (3, 2, 0.30), (3, 3, 0.50),
                ] {
                    rect {
                        x: "{23.5 + c as f64 * 10.5:.1}", y: "{5.5 + r as f64 * 10.5:.1}",
                        width: "9.2", height: "9.2", rx: "1",
                        fill: "var(--ur-azure)", fill_opacity: "{op}",
                    }
                }
                // Gold permutation matching highlights
                for (r, c) in [(0, 0), (1, 2), (2, 3), (3, 1)] {
                    rect {
                        x: "{23.5 + c as f64 * 10.5:.1}", y: "{5.5 + r as f64 * 10.5:.1}",
                        width: "9.2", height: "9.2", rx: "1",
                        fill: "none", stroke: "var(--dimer-z)", stroke_width: "1.8",
                    }
                }
            }
        },

        // Random Sparse Matrix: varied sparse blue entries + gold permutation cells
        (_, ProblemSpec::RandomMatrix { .. }) => rsx! {
            svg { class: "preset-diagram", view_box: "0 0 88 52", fill: "none",
                rect { x: "22", y: "4", width: "44", height: "44", rx: "2", fill: "var(--bg-subtle)", stroke: "var(--border-strong)", stroke_width: "1.1" }
                for (r, c, op) in [
                    (0, 2, 0.88), (0, 4, 0.32),
                    (1, 0, 0.76), (1, 3, 0.52),
                    (2, 1, 0.40), (2, 4, 0.94),
                    (3, 1, 0.85), (3, 2, 0.28),
                    (4, 0, 0.42), (4, 3, 0.90),
                ] {
                    rect {
                        x: "{23.8 + c as f64 * 8.3:.1}", y: "{5.8 + r as f64 * 8.3:.1}",
                        width: "7.0", height: "7.0", rx: "1",
                        fill: "var(--ur-azure)", fill_opacity: "{op}",
                    }
                }
                for (r, c) in [(0, 2), (1, 0), (2, 4), (3, 1), (4, 3)] {
                    rect {
                        x: "{23.8 + c as f64 * 8.3:.1}", y: "{5.8 + r as f64 * 8.3:.1}",
                        width: "7.0", height: "7.0", rx: "1",
                        fill: "none", stroke: "var(--dimer-z)", stroke_width: "1.5",
                    }
                }
            }
        },

        // Custom Matrix: crisp bracketed matrix + coordinate dots
        (_, ProblemSpec::MatrixMarket { .. }) => rsx! {
            svg { class: "preset-diagram", view_box: "0 0 88 52", fill: "none",
                // Left & right matrix brackets
                path { d: "M27 8 H21 V44 H27", stroke: "var(--dimer-x)", stroke_width: "1.8", stroke_linecap: "round", stroke_linejoin: "round" }
                path { d: "M61 8 H67 V44 H61", stroke: "var(--dimer-x)", stroke_width: "1.8", stroke_linecap: "round", stroke_linejoin: "round" }
                // Sparse coordinate entries
                circle { cx: "32", cy: "15", r: "2.8", fill: "var(--dimer-x)" }
                circle { cx: "44", cy: "15", r: "2.2", fill: "var(--ur-azure)", fill_opacity: "0.45" }
                circle { cx: "56", cy: "26", r: "2.8", fill: "var(--dimer-z)" }
                circle { cx: "32", cy: "26", r: "2.2", fill: "var(--ur-azure)", fill_opacity: "0.45" }
                circle { cx: "44", cy: "37", r: "2.8", fill: "var(--dimer-x)" }
                circle { cx: "56", cy: "37", r: "2.2", fill: "var(--dimer-z)", fill_opacity: "0.55" }
            }
        },
    }
}

#[component]
pub fn PresetStrip(
    presets: Vec<PresetInfo>,
    active_preset_id: Option<String>,
    on_select_preset: EventHandler<PresetInfo>,
    on_select_custom_mtx: EventHandler<()>,
) -> Element {
    let custom_active = active_preset_id.as_deref() == Some("custom-mtx");

    rsx! {
        div { class: "preset-strip", role: "tablist", "aria-label": "Demo problems",
            for preset in presets {
                {
                    let is_active = active_preset_id.as_deref() == Some(preset.id.as_str());
                    let (short_title, size_tag) = short_card_labels(&preset);
                    let tooltip = format!("{} — {}", preset.title, preset.subtitle);
                    let preset_clone = preset.clone();
                    rsx! {
                        button {
                            key: "{preset.id}",
                            r#type: "button",
                            role: "tab",
                            title: "{tooltip}",
                            "aria-selected": is_active,
                            class: if is_active { "preset-card active" } else { "preset-card" },
                            onclick: move |_| on_select_preset.call(preset_clone.clone()),
                            PresetDiagram { id: preset.id.clone(), spec: preset.spec.clone() }
                            div { class: "preset-footer",
                                span { class: "preset-title", "{short_title}" }
                                if !size_tag.is_empty() {
                                    span { class: "preset-size mono", "{size_tag}" }
                                }
                            }
                        }
                    }
                }
            }

            button {
                key: "custom-mtx",
                r#type: "button",
                role: "tab",
                title: "Paste or edit any sparse coordinate matrix (n ≤ 1024)",
                "aria-selected": custom_active,
                class: if custom_active { "preset-card active" } else { "preset-card" },
                onclick: move |_| on_select_custom_mtx.call(()),
                PresetDiagram {
                    id: "custom-mtx",
                    spec: ProblemSpec::MatrixMarket { text: String::new() },
                }
                div { class: "preset-footer",
                    span { class: "preset-title", "Custom Matrix" }
                    span { class: "preset-size mono", "custom" }
                }
            }
        }
    }
}
