//! Interactive log-scale weighted matrix heatmap with highlighted active
//! perfect matching cells and hover inspection.

use std::collections::HashSet;

use dioxus::prelude::*;
use permanent_demo::types::MatrixEntry;

use crate::state::format_sig;

#[component]
pub fn ViewerMatrix(n: usize, triplets: Vec<MatrixEntry>, matching: Option<Vec<u32>>) -> Element {
    let mut hovered = use_signal(|| None::<(u32, u32, f64, bool)>);

    let mut min_w = f64::INFINITY;
    let mut max_w = 0.0_f64;
    for t in &triplets {
        if t.value > 0.0 {
            min_w = min_w.min(t.value);
            max_w = max_w.max(t.value);
        }
    }
    let min_ln = if min_w.is_finite() { min_w.ln() } else { 0.0 };
    let max_ln = if max_w > 0.0 { max_w.ln() } else { 0.0 };
    let span = (max_ln - min_ln).max(1e-9);

    let matched_pairs: HashSet<(u32, u32)> =
        matching.as_ref().map(|m| m.iter().enumerate().map(|(u, &v)| (u as u32, v)).collect()).unwrap_or_default();

    let board = 420.0_f64;
    let pad = 28.0_f64;
    let cell = board / n.max(1) as f64;
    let vb = board + 2.0 * pad;
    let view_box = format!("0 0 {vb:.1} {vb:.1}");

    // Cap per-cell DOM nodes on very large matrices by still rendering all up to n=64 cleanly;
    // for n > 64 render downsampled / sparse dots cleanly.
    let sparse_mode = triplets.len() > 4500;

    rsx! {
        div { class: "viewport-body",
            if let Some((u, v, val, is_m)) = hovered() {
                div { class: "matrix-tooltip",
                    "A[{u + 1}, {v + 1}] = {format_sig(val)}"
                    if is_m { " · matched" }
                }
            }

            div { class: "svg-stage",
                svg { view_box: "{view_box}", preserve_aspect_ratio: "xMidYMid meet",
                    // Matrix background frame
                    rect {
                        x: "{pad:.1}",
                        y: "{pad:.1}",
                        width: "{board:.1}",
                        height: "{board:.1}",
                        fill: "var(--bg-subtle)",
                        stroke: "var(--border-strong)",
                        stroke_width: "1.4",
                    }

                    // Axis annotations
                    text {
                        x: "{pad + board * 0.5:.1}",
                        y: "{pad - 9.0:.1}",
                        text_anchor: "middle",
                        fill: "var(--text-muted)",
                        font_size: "11",
                        "columns 1 to {n}"
                    }
                    text {
                        x: "{pad - 10.0:.1}",
                        y: "{pad + board * 0.5:.1}",
                        text_anchor: "middle",
                        fill: "var(--text-muted)",
                        font_size: "11",
                        transform: "rotate(-90 {pad - 10.0:.1} {pad + board * 0.5:.1})",
                        "rows 1 to {n}"
                    }

                    // Nonzero cells shaded on a logarithmic blue scale
                    if !sparse_mode {
                        for entry in &triplets {
                            {
                                let t = if (max_ln - min_ln).abs() < 1e-12 {
                                    0.72
                                } else {
                                    ((entry.value.ln() - min_ln) / span).clamp(0.0, 1.0)
                                };
                                let alpha = 0.18 + 0.80 * t;
                                let x = pad + entry.col as f64 * cell;
                                let y = pad + entry.row as f64 * cell;
                                let r = entry.row;
                                let c = entry.col;
                                let val = entry.value;
                                let is_m = matched_pairs.contains(&(r, c));
                                rsx! {
                                    rect {
                                        x: "{x:.2}",
                                        y: "{y:.2}",
                                        width: "{(cell - 0.5).max(1.0):.2}",
                                        height: "{(cell - 0.5).max(1.0):.2}",
                                        fill: "var(--ur-azure)",
                                        fill_opacity: "{alpha:.2}",
                                        onmouseenter: move |_| hovered.set(Some((r, c, val, is_m))),
                                        onmouseleave: move |_| hovered.set(None),
                                    }
                                }
                            }
                        }
                    } else {
                        text {
                            x: "{pad + board * 0.5:.1}",
                            y: "{pad + board * 0.5:.1}",
                            text_anchor: "middle",
                            fill: "var(--text-muted)",
                            font_size: "13",
                            "{n} × {n} dense matrix ({triplets.len()} nonzeros) · displaying active matching"
                        }
                    }

                    // Highlight active matching entries in crisp gold borders
                    if let Some(ref m) = matching {
                        for (u, &v) in m.iter().enumerate() {
                            {
                                let x = pad + v as f64 * cell;
                                let y = pad + u as f64 * cell;
                                let sw = if n <= 24 { 2.4 } else { 1.4 };
                                rsx! {
                                    rect {
                                        x: "{x + 0.6:.2}",
                                        y: "{y + 0.6:.2}",
                                        width: "{(cell - 1.4).max(1.8):.2}",
                                        height: "{(cell - 1.4).max(1.8):.2}",
                                        fill: "none",
                                        stroke: "var(--dimer-z)",
                                        stroke_width: "{sw:.1}",
                                    }
                                }
                            }
                        }
                    }
                }
            }

            div { class: "hud-legend",
                span { class: "legend-item",
                    span { class: "swatch-rod", style: "background: var(--ur-azure);" }
                    "log weight"
                }
                span { class: "legend-item",
                    span {
                        style: "width: 10px; height: 10px; border: 2px solid var(--dimer-z); display: inline-block;",
                    }
                    "active matching"
                }
            }
        }
    }
}
