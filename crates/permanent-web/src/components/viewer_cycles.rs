//! Disjoint bipartite cycles diagram showing each cycle's two complementary
//! perfect matchings (total permanent = 2^k).

use std::f64::consts::TAU;

use dioxus::prelude::*;
use permanent_demo::types::ProblemGeometry;

#[component]
pub fn ViewerCycles(geometry: ProblemGeometry, matching: Option<Vec<u32>>) -> Element {
    let ProblemGeometry::Cycles { cycle_sizes } = geometry else {
        return rsx! { div {} };
    };

    let k = cycle_sizes.len().max(1);
    let cols_grid = k.min(4);
    let rows_grid = k.div_ceil(cols_grid);
    let tile = 210.0_f64;
    let vb_w = cols_grid as f64 * tile;
    let vb_h = rows_grid as f64 * tile;
    let view_box = format!("0 0 {vb_w:.1} {vb_h:.1}");

    let mut offsets = Vec::with_capacity(k);
    let mut acc = 0usize;
    for &sz in &cycle_sizes {
        offsets.push(acc);
        acc += sz;
    }

    rsx! {
        div { class: "viewport-body",
            div { class: "svg-stage",
                svg { view_box: "{view_box}", preserve_aspect_ratio: "xMidYMid meet",
                    for (ci, &size) in cycle_sizes.iter().enumerate() {
                        {
                            let offset = offsets[ci];
                            let gx = ci % cols_grid;
                            let gy = ci / cols_grid;
                            let cx = (gx as f64 + 0.5) * tile;
                            let cy = (gy as f64 + 0.5) * tile;
                            let r = 74.0_f64;
                            let total_nodes = 2 * size;
                            let node_xy = |idx: usize| -> (f64, f64) {
                                let angle = -std::f64::consts::FRAC_PI_2 + TAU * (idx as f64 / total_nodes as f64);
                                (cx + r * angle.cos(), cy + r * angle.sin())
                            };

                            // Inspect which choice this cycle is currently in.
                            let choice_label = match &matching {
                                Some(m) if offset < m.len() => {
                                    let matched_col = m[offset] as usize;
                                    if matched_col == offset {
                                        "choice 1"
                                    } else if matched_col == offset + (1 % size) {
                                        "choice 2"
                                    } else {
                                        "mixed"
                                    }
                                }
                                _ => "2 matchings",
                            };
                            let active_stroke = if choice_label == "choice 2" {
                                "var(--dimer-z)"
                            } else {
                                "var(--dimer-x)"
                            };

                            rsx! {
                                g { key: "cycle-{ci}",
                                    // Cycle outline edges
                                    for i in 0..total_nodes {
                                        {
                                            let (x1, y1) = node_xy(i);
                                            let (x2, y2) = node_xy((i + 1) % total_nodes);
                                            rsx! {
                                                line {
                                                    x1: "{x1:.1}",
                                                    y1: "{y1:.1}",
                                                    x2: "{x2:.1}",
                                                    y2: "{y2:.1}",
                                                    stroke: "var(--lattice-wire)",
                                                    stroke_width: "1.6",
                                                }
                                            }
                                        }
                                    }

                                    // Active matched edges
                                    if let Some(ref m) = matching {
                                        for i in 0..size {
                                            {
                                                let u = offset + i;
                                                if let Some(&v_u32) = m.get(u) {
                                                    let v = v_u32 as usize;
                                                    if v >= offset && v < offset + size {
                                                        let col_local = v - offset;
                                                        let row_pos = 2 * i;
                                                        let col_pos = 2 * col_local + 1;
                                                        if col_pos == (row_pos + 1) % total_nodes
                                                            || (col_pos + 1) % total_nodes == row_pos
                                                        {
                                                            let (x1, y1) = node_xy(row_pos);
                                                            let (x2, y2) = node_xy(col_pos);
                                                            rsx! {
                                                                line {
                                                                    x1: "{x1:.1}",
                                                                    y1: "{y1:.1}",
                                                                    x2: "{x2:.1}",
                                                                    y2: "{y2:.1}",
                                                                    stroke: "{active_stroke}",
                                                                    stroke_width: "4.6",
                                                                    stroke_linecap: "round",
                                                                }
                                                            }
                                                        } else {
                                                            rsx! {}
                                                        }
                                                    } else {
                                                        rsx! {}
                                                    }
                                                } else {
                                                    rsx! {}
                                                }
                                            }
                                        }
                                    }

                                    // Alternating bipartite vertices (● row / ○ col)
                                    if total_nodes <= 64 {
                                        for i in 0..total_nodes {
                                            {
                                                let (nx, ny) = node_xy(i);
                                                let is_row = i % 2 == 0;
                                                let nr = if total_nodes <= 32 { 3.6 } else { 2.4 };
                                                rsx! {
                                                    circle {
                                                        cx: "{nx:.1}",
                                                        cy: "{ny:.1}",
                                                        r: "{nr:.1}",
                                                        fill: if is_row { "var(--node-row-fill)" } else { "var(--node-col-fill)" },
                                                        stroke: if is_row { "none" } else { "var(--node-col-stroke)" },
                                                        stroke_width: "1.3",
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Center label
                                    text {
                                        x: "{cx:.1}",
                                        y: "{cy - 5.0:.1}",
                                        text_anchor: "middle",
                                        fill: "var(--text-primary)",
                                        font_size: "12.5",
                                        font_weight: "600",
                                        "cycle {ci + 1}"
                                    }
                                    text {
                                        x: "{cx:.1}",
                                        y: "{cy + 11.0:.1}",
                                        text_anchor: "middle",
                                        fill: "{active_stroke}",
                                        font_size: "11",
                                        font_weight: "600",
                                        "{choice_label} ({size}×{size})"
                                    }
                                }
                            }
                        }
                    }
                }
            }

            div { class: "hud-legend",
                span { class: "legend-item",
                    span { class: "swatch-dot-solid" }
                    "row"
                }
                span { class: "legend-item",
                    span { class: "swatch-dot-hollow" }
                    "col"
                }
                span { class: "legend-item",
                    span { class: "swatch-rod", style: "background: var(--dimer-x);" }
                    "choice 1"
                }
                span { class: "legend-item",
                    span { class: "swatch-rod", style: "background: var(--dimer-z);" }
                    "choice 2"
                }
            }
        }
    }
}
