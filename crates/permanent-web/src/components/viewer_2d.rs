//! Interactive 2D square/rectangular lattice visualizer supporting both
//! weighted bipartite bond view and classic domino tiling view.

use std::collections::HashMap;

use dioxus::prelude::*;
use permanent_demo::types::{LatticeBond, ProblemGeometry};

use crate::state::Viewer2dMode;

#[component]
pub fn Viewer2d(geometry: ProblemGeometry, matching: Option<Vec<u32>>, is_zero_one: bool) -> Element {
    let mut mode = use_signal(move || if is_zero_one { Viewer2dMode::Dominoes } else { Viewer2dMode::Bonds });
    let mut zoom = use_signal(|| 1.0_f64);

    let ProblemGeometry::Lattice2d { rows, cols, periodic, row_sites, col_sites, bonds } = geometry else {
        return rsx! { div {} };
    };

    // Index bonds by (row, col) for O(1) active-dimer lookup.
    let mut bond_map: HashMap<(usize, usize), &LatticeBond> = HashMap::with_capacity(bonds.len());
    for b in &bonds {
        let key = (b.row as usize, b.col as usize);
        if !bond_map.contains_key(&key) || !b.periodic_wrap {
            bond_map.insert(key, b);
        }
    }

    let cell = 42.0_f64;
    let pad = 28.0_f64;
    let board_w = cols as f64 * cell;
    let board_h = rows as f64 * cell;
    let vb_w = board_w + 2.0 * pad;
    let vb_h = board_h + 2.0 * pad;

    let z = zoom();
    let scaled_w = vb_w / z;
    let scaled_h = vb_h / z;
    let vx = (vb_w - scaled_w) * 0.5;
    let vy = (vb_h - scaled_h) * 0.5;
    let view_box = format!("{vx:.1} {vy:.1} {scaled_w:.1} {scaled_h:.1}");

    let center = |x: usize, y: usize| -> (f64, f64) { (pad + (y as f64 + 0.5) * cell, pad + (x as f64 + 0.5) * cell) };
    let active_mode = mode();
    let show_nodes = rows * cols <= 400;

    rsx! {
        div { class: "viewport-body",
            div { class: "svg-stage",
                svg { view_box: "{view_box}", preserve_aspect_ratio: "xMidYMid meet",
                    if active_mode == Viewer2dMode::Dominoes {
                        // Subtle checkerboard background
                        for x in 0..rows {
                            for y in 0..cols {
                                {
                                    let px = pad + y as f64 * cell;
                                    let py = pad + x as f64 * cell;
                                    let dark = (x + y) % 2 == 0;
                                    rsx! {
                                        rect {
                                            x: "{px:.1}",
                                            y: "{py:.1}",
                                            width: "{cell:.1}",
                                            height: "{cell:.1}",
                                            fill: if dark { "var(--bg-subtle)" } else { "var(--bg-surface)" },
                                            stroke: "var(--border-hairline)",
                                            stroke_width: "0.8",
                                        }
                                    }
                                }
                            }
                        }

                        // Rounded domino tiles straddling matched pairs
                        if let Some(ref m) = matching {
                            for (u, &v_u32) in m.iter().enumerate() {
                                {
                                    let v = v_u32 as usize;
                                    let Some(&[rx, ry]) = row_sites.get(u) else { return rsx! {}; };
                                    let Some(&[cx, cy]) = col_sites.get(v) else { return rsx! {}; };
                                    let bond = bond_map.get(&(u, v)).copied();
                                    let inset = 4.5_f64;
                                    if let Some(b) = bond && !b.periodic_wrap {
                                        let min_x = ry.min(cy) as f64;
                                        let max_x = ry.max(cy) as f64;
                                        let min_y = rx.min(cx) as f64;
                                        let max_y = rx.max(cx) as f64;
                                        let x0 = pad + min_x * cell + inset;
                                        let y0 = pad + min_y * cell + inset;
                                        let w = (max_x - min_x + 1.0) * cell - 2.0 * inset;
                                        let h = (max_y - min_y + 1.0) * cell - 2.0 * inset;
                                        let horizontal = rx == cx;
                                        let stroke = if horizontal { "var(--dimer-x)" } else { "var(--dimer-z)" };
                                        let fill = if horizontal { "var(--accent-blue-tint)" } else { "var(--accent-gold-tint)" };
                                        rsx! {
                                            rect {
                                                x: "{x0:.1}",
                                                y: "{y0:.1}",
                                                width: "{w:.1}",
                                                height: "{h:.1}",
                                                rx: "7",
                                                fill: "{fill}",
                                                stroke: "{stroke}",
                                                stroke_width: "2.2",
                                            }
                                        }
                                    } else {
                                        // Periodic wrap or off-support draw: draw crisp line indicator
                                        let (x1, y1) = center(rx, ry);
                                        let (x2, y2) = center(cx, cy);
                                        rsx! {
                                            line {
                                                x1: "{x1:.1}",
                                                y1: "{y1:.1}",
                                                x2: "{x2:.1}",
                                                y2: "{y2:.1}",
                                                stroke: if bond.is_some() { "var(--dimer-z)" } else { "var(--status-err)" },
                                                stroke_width: "2.4",
                                                stroke_dasharray: "5 4",
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Outer board frame
                        rect {
                            x: "{pad:.1}",
                            y: "{pad:.1}",
                            width: "{board_w:.1}",
                            height: "{board_h:.1}",
                            fill: "none",
                            stroke: "var(--border-strong)",
                            stroke_width: "1.6",
                        }
                    } else {
                        // Bonds mode: hairline lattice edges (dashed on vertical/light bonds)
                        for b in &bonds {
                            if !b.periodic_wrap {
                                {
                                    let [rx, ry] = row_sites[b.row as usize];
                                    let [cx, cy] = col_sites[b.col as usize];
                                    let (x1, y1) = center(rx, ry);
                                    let (x2, y2) = center(cx, cy);
                                    rsx! {
                                        line {
                                            x1: "{x1:.1}",
                                            y1: "{y1:.1}",
                                            x2: "{x2:.1}",
                                            y2: "{y2:.1}",
                                            stroke: "var(--lattice-wire)",
                                            stroke_width: "1.3",
                                            stroke_dasharray: if b.axis == 1 { "4 3" } else { "" },
                                        }
                                    }
                                }
                            }
                        }

                        // Active dimers as crisp rounded rods
                        if let Some(ref m) = matching {
                            for (u, &v_u32) in m.iter().enumerate() {
                                {
                                    let v = v_u32 as usize;
                                    let Some(&[rx, ry]) = row_sites.get(u) else { return rsx! {}; };
                                    let Some(&[cx, cy]) = col_sites.get(v) else { return rsx! {}; };
                                    let bond = bond_map.get(&(u, v)).copied();
                                    let (x1, y1) = center(rx, ry);
                                    let (x2, y2) = center(cx, cy);
                                    let stroke = match bond {
                                        Some(b) if b.axis == 0 => "var(--dimer-x)",
                                        Some(_) => "var(--dimer-z)",
                                        None => "var(--status-err)",
                                    };
                                    let dashed = bond.map(|b| b.periodic_wrap).unwrap_or(true);
                                    rsx! {
                                        line {
                                            x1: "{x1:.1}",
                                            y1: "{y1:.1}",
                                            x2: "{x2:.1}",
                                            y2: "{y2:.1}",
                                            stroke: "{stroke}",
                                            stroke_width: if dashed { "3.0" } else { "6.5" },
                                            stroke_linecap: "round",
                                            stroke_dasharray: if dashed { "6 4" } else { "" },
                                        }
                                    }
                                }
                            }
                        }

                        // Bipartite site markers (● row / ○ col)
                        if show_nodes {
                            for &[rx, ry] in &row_sites {
                                {
                                    let (px, py) = center(rx, ry);
                                    rsx! {
                                        circle {
                                            cx: "{px:.1}",
                                            cy: "{py:.1}",
                                            r: "4.2",
                                            fill: "var(--node-row-fill)",
                                        }
                                    }
                                }
                            }
                            for &[cx, cy] in &col_sites {
                                {
                                    let (px, py) = center(cx, cy);
                                    rsx! {
                                        circle {
                                            cx: "{px:.1}",
                                            cy: "{py:.1}",
                                            r: "4.2",
                                            fill: "var(--node-col-fill)",
                                            stroke: "var(--node-col-stroke)",
                                            stroke_width: "1.6",
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            div { class: "hud-bar",
                button {
                    class: if active_mode == Viewer2dMode::Bonds { "hud-btn active" } else { "hud-btn" },
                    r#type: "button",
                    onclick: move |_| mode.set(Viewer2dMode::Bonds),
                    "Bonds"
                }
                button {
                    class: if active_mode == Viewer2dMode::Dominoes { "hud-btn active" } else { "hud-btn" },
                    r#type: "button",
                    onclick: move |_| mode.set(Viewer2dMode::Dominoes),
                    "Dominoes"
                }
                span { class: "hud-sep" }
                button {
                    class: "hud-btn",
                    r#type: "button",
                    onclick: move |_| zoom.set((zoom() / 1.25).max(0.75)),
                    "−"
                }
                span { class: "mono", style: "font-size: 0.72rem; padding: 0 0.2rem;", "{(z * 100.0).round() as i32}%" }
                button {
                    class: "hud-btn",
                    r#type: "button",
                    onclick: move |_| zoom.set((zoom() * 1.25).min(3.0)),
                    "+"
                }
                button {
                    class: "hud-btn",
                    r#type: "button",
                    onclick: move |_| zoom.set(1.0),
                    "Reset"
                }
            }

            div { class: "hud-legend",
                span { class: "legend-item",
                    span { class: "swatch-rod", style: "background: var(--dimer-x);" }
                    "axis 1"
                }
                span { class: "legend-item",
                    span { class: "swatch-rod", style: "background: var(--dimer-z);" }
                    "axis 2"
                }
                if periodic {
                    span { class: "legend-item", "periodic" }
                }
            }
        }
    }
}
