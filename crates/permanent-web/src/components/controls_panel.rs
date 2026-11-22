//! Problem parameter builder and two-tier WebGPU run configuration panel.

use dioxus::prelude::*;
use permanent_demo::types::{ProblemSpec, RandomMatrixFamily};
use permanent_wasm::types::WebRunConfig;

const BUNDLED_PRESETS: &[(&str, &str)] = &[
    ("weighted-16", "16 × 16 Weighted Matrix"),
    ("weighted-8", "8 × 8 Weighted Matrix"),
    ("weighted-12", "12 × 12 Weighted Matrix"),
    ("weighted-20", "20 × 20 Sparse Weighted Matrix"),
    ("weighted-blocks-48", "48 × 48 Coupled Weighted Blocks"),
    ("dimer-8x8", "8 × 8 Weighted Square Dimers"),
    ("dimer-16x16", "16 × 16 Weighted Square Dimers"),
    ("dimer-torus-6x6x6", "6 × 6 × 6 Periodic Cubic Torus"),
];

const RANDOM_FAMILIES: &[(RandomMatrixFamily, &str)] = &[
    (RandomMatrixFamily::SparseLognormal, "Sparse Lognormal"),
    (RandomMatrixFamily::ZeroOneBipartite, "0/1 Bipartite"),
    (RandomMatrixFamily::NearPermutation, "Near-Permutation"),
    (RandomMatrixFamily::WeaklyCoupledBlocks, "Coupled Blocks"),
    (RandomMatrixFamily::Uniform, "Uniform"),
];

pub const SAMPLE_MTX_WEIGHTED: &str = "%%MatrixMarket matrix coordinate real general\n\
4 4 12\n\
1 1 0.9\n1 2 0.2\n1 4 0.4\n\
2 1 0.1\n2 2 0.7\n2 3 0.5\n\
3 2 0.3\n3 3 0.8\n3 4 0.6\n\
4 1 0.5\n4 3 0.2\n4 4 0.9\n";

pub const SAMPLE_MTX_ZERO: &str = "%%MatrixMarket matrix coordinate real general\n\
3 3 4\n\
1 1 1\n2 1 2\n3 1 1\n3 2 1\n";

fn parse_opt_usize(raw: &str) -> Option<usize> {
    let trimmed = raw.trim();
    if trimmed.is_empty() { None } else { trimmed.parse::<usize>().ok().filter(|&v| v > 0) }
}

fn parse_opt_u32(raw: &str) -> Option<u32> {
    let trimmed = raw.trim();
    if trimmed.is_empty() { None } else { trimmed.parse::<u32>().ok().filter(|&v| v > 0) }
}

fn next_seed(current: u64) -> u64 {
    current.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) % 99_991 + 1
}

#[component]
pub fn ControlsPanel(
    spec: ProblemSpec,
    config: WebRunConfig,
    running: bool,
    on_change_spec: EventHandler<ProblemSpec>,
    on_change_config: EventHandler<WebRunConfig>,
) -> Element {
    let mut show_advanced = use_signal(|| false);

    rsx! {
        div { class: "panel-scroll",
            // ========================================================
            // Section 1 — Problem Parameters
            // ========================================================
            div { class: "control-group",
                h4 { class: "control-group-title",
                    span { "Problem Parameters" }
                }

                match spec.clone() {
                    ProblemSpec::Dimer3d {
                        lx,
                        ly,
                        lz,
                        periodic,
                        weights,
                        disorder,
                        vacancy_rate,
                        ensure_matching,
                        seed,
                    } => {
                        let update = move |nlx, nly, nlz, np, nw, nd, nv, nem, ns| {
                            on_change_spec.call(ProblemSpec::Dimer3d {
                                lx: nlx,
                                ly: nly,
                                lz: nlz,
                                periodic: np,
                                weights: nw,
                                disorder: nd,
                                vacancy_rate: nv,
                                ensure_matching: nem,
                                seed: ns,
                            });
                        };
                        rsx! {
                            div { class: "field-grid-3",
                                div { class: "field",
                                    label { "Lx" }
                                    input {
                                        r#type: "number", min: "2", max: "16", step: if periodic { "2" } else { "1" },
                                        disabled: running,
                                        value: "{lx}",
                                        onchange: move |ev| {
                                            if let Ok(v) = ev.value().parse::<usize>() {
                                                update(v.clamp(2, 16), ly, lz, periodic, weights, disorder, vacancy_rate, ensure_matching, seed);
                                            }
                                        },
                                    }
                                }
                                div { class: "field",
                                    label { "Ly" }
                                    input {
                                        r#type: "number", min: "2", max: "16", step: if periodic { "2" } else { "1" },
                                        disabled: running,
                                        value: "{ly}",
                                        onchange: move |ev| {
                                            if let Ok(v) = ev.value().parse::<usize>() {
                                                update(lx, v.clamp(2, 16), lz, periodic, weights, disorder, vacancy_rate, ensure_matching, seed);
                                            }
                                        },
                                    }
                                }
                                div { class: "field",
                                    label { "Lz" }
                                    input {
                                        r#type: "number", min: "2", max: "16", step: if periodic { "2" } else { "1" },
                                        disabled: running,
                                        value: "{lz}",
                                        onchange: move |ev| {
                                            if let Ok(v) = ev.value().parse::<usize>() {
                                                update(lx, ly, v.clamp(2, 16), periodic, weights, disorder, vacancy_rate, ensure_matching, seed);
                                            }
                                        },
                                    }
                                }
                            }

                            div { class: "field",
                                label { "Boundary Condition" }
                                div { class: "segmented",
                                    button {
                                        r#type: "button",
                                        disabled: running,
                                        class: if !periodic { "active" } else { "" },
                                        onclick: move |_| update(lx, ly, lz, false, weights, disorder, vacancy_rate, ensure_matching, seed),
                                        "Open Box"
                                    }
                                    button {
                                        r#type: "button",
                                        disabled: running,
                                        class: if periodic { "active" } else { "" },
                                        onclick: move |_| {
                                            let even = |d: usize| (d.max(4) + (d.max(4) % 2)).min(12);
                                            update(even(lx), even(ly), even(lz), true, weights, disorder, vacancy_rate, ensure_matching, seed);
                                        },
                                        "Periodic Torus"
                                    }
                                }
                            }

                            div { class: "field-grid-3",
                                div { class: "field",
                                    label { "Weight X" }
                                    input {
                                        r#type: "number", min: "0", max: "100", step: "0.1",
                                        disabled: running,
                                        value: "{weights[0]}",
                                        onchange: move |ev| {
                                            if let Ok(v) = ev.value().parse::<f64>() {
                                                update(lx, ly, lz, periodic, [v.max(0.0), weights[1], weights[2]], disorder, vacancy_rate, ensure_matching, seed);
                                            }
                                        },
                                    }
                                }
                                div { class: "field",
                                    label { "Weight Y" }
                                    input {
                                        r#type: "number", min: "0", max: "100", step: "0.1",
                                        disabled: running,
                                        value: "{weights[1]}",
                                        onchange: move |ev| {
                                            if let Ok(v) = ev.value().parse::<f64>() {
                                                update(lx, ly, lz, periodic, [weights[0], v.max(0.0), weights[2]], disorder, vacancy_rate, ensure_matching, seed);
                                            }
                                        },
                                    }
                                }
                                div { class: "field",
                                    label { "Weight Z" }
                                    input {
                                        r#type: "number", min: "0", max: "100", step: "0.1",
                                        disabled: running,
                                        value: "{weights[2]}",
                                        onchange: move |ev| {
                                            if let Ok(v) = ev.value().parse::<f64>() {
                                                update(lx, ly, lz, periodic, [weights[0], weights[1], v.max(0.0)], disorder, vacancy_rate, ensure_matching, seed);
                                            }
                                        },
                                    }
                                }
                            }

                            div { class: "field-grid-2",
                                div { class: "field",
                                    label {
                                        span { "Bond Disorder" }
                                        span { class: "mono", "±{disorder:.2}" }
                                    }
                                    input {
                                        r#type: "range", min: "0", max: "4", step: "0.1",
                                        disabled: running,
                                        value: "{disorder}",
                                        oninput: move |ev| {
                                            if let Ok(v) = ev.value().parse::<f64>() {
                                                update(lx, ly, lz, periodic, weights, v, vacancy_rate, ensure_matching, seed);
                                            }
                                        },
                                    }
                                }
                                div { class: "field",
                                    label {
                                        span { "Vacancy Rate" }
                                        span { class: "mono", "{(vacancy_rate * 100.0).round() as i32}%" }
                                    }
                                    input {
                                        r#type: "range", min: "0", max: "0.6", step: "0.05",
                                        disabled: running,
                                        value: "{vacancy_rate}",
                                        oninput: move |ev| {
                                            if let Ok(v) = ev.value().parse::<f64>() {
                                                update(lx, ly, lz, periodic, weights, disorder, v, ensure_matching, seed);
                                            }
                                        },
                                    }
                                }
                            }

                            div { class: "field-grid-2",
                                label { class: "toggle-item",
                                    input {
                                        r#type: "checkbox",
                                        disabled: running,
                                        checked: ensure_matching,
                                        onchange: move |ev| {
                                            update(lx, ly, lz, periodic, weights, disorder, vacancy_rate, ev.checked(), seed);
                                        },
                                    }
                                    "Preserve matching"
                                }
                                button {
                                    class: "btn-secondary",
                                    r#type: "button",
                                    disabled: running,
                                    onclick: move |_| {
                                        let next = next_seed(seed);
                                        let d = if disorder == 0.0 && vacancy_rate == 0.0 { 0.8 } else { disorder };
                                        update(lx, ly, lz, periodic, weights, d, vacancy_rate, ensure_matching, next);
                                    },
                                    "Randomize Lattice"
                                }
                            }
                        }
                    }

                    ProblemSpec::Dimer2d {
                        rows,
                        cols,
                        periodic,
                        weight_x,
                        weight_y,
                        disorder,
                        vacancy_rate,
                        ensure_matching,
                        seed,
                    } => {
                        let update = move |nr, nc, np, wx, wy, nd, nv, nem, ns| {
                            on_change_spec.call(ProblemSpec::Dimer2d {
                                rows: nr,
                                cols: nc,
                                periodic: np,
                                weight_x: wx,
                                weight_y: wy,
                                disorder: nd,
                                vacancy_rate: nv,
                                ensure_matching: nem,
                                seed: ns,
                            });
                        };
                        rsx! {
                            div { class: "field-grid-2",
                                div { class: "field",
                                    label { "Rows" }
                                    input {
                                        r#type: "number", min: "2", max: "44", step: "2",
                                        disabled: running,
                                        value: "{rows}",
                                        onchange: move |ev| {
                                            if let Ok(v) = ev.value().parse::<usize>() {
                                                update(v.clamp(2, 44), cols, periodic, weight_x, weight_y, disorder, vacancy_rate, ensure_matching, seed);
                                            }
                                        },
                                    }
                                }
                                div { class: "field",
                                    label { "Columns" }
                                    input {
                                        r#type: "number", min: "2", max: "44", step: "2",
                                        disabled: running,
                                        value: "{cols}",
                                        onchange: move |ev| {
                                            if let Ok(v) = ev.value().parse::<usize>() {
                                                update(rows, v.clamp(2, 44), periodic, weight_x, weight_y, disorder, vacancy_rate, ensure_matching, seed);
                                            }
                                        },
                                    }
                                }
                            }

                            div { class: "field",
                                label { "Boundary Condition" }
                                div { class: "segmented",
                                    button {
                                        r#type: "button",
                                        disabled: running,
                                        class: if !periodic { "active" } else { "" },
                                        onclick: move |_| update(rows, cols, false, weight_x, weight_y, disorder, vacancy_rate, ensure_matching, seed),
                                        "Open Board"
                                    }
                                    button {
                                        r#type: "button",
                                        disabled: running,
                                        class: if periodic { "active" } else { "" },
                                        onclick: move |_| {
                                            let even = |d: usize| (d.max(4) + (d.max(4) % 2)).min(44);
                                            update(even(rows), even(cols), true, weight_x, weight_y, disorder, vacancy_rate, ensure_matching, seed);
                                        },
                                        "Periodic Torus"
                                    }
                                }
                            }

                            div { class: "field-grid-2",
                                div { class: "field",
                                    label { "Weight X" }
                                    input {
                                        r#type: "number", min: "0", max: "100", step: "0.1",
                                        disabled: running,
                                        value: "{weight_x}",
                                        onchange: move |ev| {
                                            if let Ok(v) = ev.value().parse::<f64>() {
                                                update(rows, cols, periodic, v.max(0.0), weight_y, disorder, vacancy_rate, ensure_matching, seed);
                                            }
                                        },
                                    }
                                }
                                div { class: "field",
                                    label { "Weight Y" }
                                    input {
                                        r#type: "number", min: "0", max: "100", step: "0.1",
                                        disabled: running,
                                        value: "{weight_y}",
                                        onchange: move |ev| {
                                            if let Ok(v) = ev.value().parse::<f64>() {
                                                update(rows, cols, periodic, weight_x, v.max(0.0), disorder, vacancy_rate, ensure_matching, seed);
                                            }
                                        },
                                    }
                                }
                            }

                            div { class: "field-grid-2",
                                div { class: "field",
                                    label {
                                        span { "Bond Disorder" }
                                        span { class: "mono", "±{disorder:.2}" }
                                    }
                                    input {
                                        r#type: "range", min: "0", max: "4", step: "0.1",
                                        disabled: running,
                                        value: "{disorder}",
                                        oninput: move |ev| {
                                            if let Ok(v) = ev.value().parse::<f64>() {
                                                update(rows, cols, periodic, weight_x, weight_y, v, vacancy_rate, ensure_matching, seed);
                                            }
                                        },
                                    }
                                }
                                div { class: "field",
                                    label {
                                        span { "Vacancy Rate" }
                                        span { class: "mono", "{(vacancy_rate * 100.0).round() as i32}%" }
                                    }
                                    input {
                                        r#type: "range", min: "0", max: "0.6", step: "0.05",
                                        disabled: running,
                                        value: "{vacancy_rate}",
                                        oninput: move |ev| {
                                            if let Ok(v) = ev.value().parse::<f64>() {
                                                update(rows, cols, periodic, weight_x, weight_y, disorder, v, ensure_matching, seed);
                                            }
                                        },
                                    }
                                }
                            }

                            div { class: "field-grid-2",
                                label { class: "toggle-item",
                                    input {
                                        r#type: "checkbox",
                                        disabled: running,
                                        checked: ensure_matching,
                                        onchange: move |ev| {
                                            update(rows, cols, periodic, weight_x, weight_y, disorder, vacancy_rate, ev.checked(), seed);
                                        },
                                    }
                                    "Preserve matching"
                                }
                                button {
                                    class: "btn-secondary",
                                    r#type: "button",
                                    disabled: running,
                                    onclick: move |_| {
                                        let next = next_seed(seed);
                                        let d = if disorder == 0.0 && vacancy_rate == 0.0 { 0.8 } else { disorder };
                                        update(rows, cols, periodic, weight_x, weight_y, d, vacancy_rate, ensure_matching, next);
                                    },
                                    "Randomize Board"
                                }
                            }
                        }
                    }

                    ProblemSpec::DisjointCycles { cycle_sizes } => {
                        let k = cycle_sizes.len();
                        let sizes_add = cycle_sizes.clone();
                        rsx! {
                            div { class: "field",
                                label {
                                    span { "Cycle Sizes (exact permanent = 2^{k})" }
                                }
                                div { class: "cycle-chips",
                                    for (idx, &sz) in cycle_sizes.iter().enumerate() {
                                        {
                                            let sizes_edit = cycle_sizes.clone();
                                            let sizes_del = cycle_sizes.clone();
                                            rsx! {
                                                div { key: "c-{idx}", class: "cycle-chip",
                                                    span { "#{idx + 1}" }
                                                    input {
                                                        r#type: "number", min: "2", max: "64",
                                                        disabled: running,
                                                        value: "{sz}",
                                                        onchange: move |ev| {
                                                            if let Ok(v) = ev.value().parse::<usize>() {
                                                                let mut next = sizes_edit.clone();
                                                                next[idx] = v.clamp(2, 64);
                                                                on_change_spec.call(ProblemSpec::DisjointCycles { cycle_sizes: next });
                                                            }
                                                        },
                                                    }
                                                    if k > 1 {
                                                        button {
                                                            r#type: "button",
                                                            disabled: running,
                                                            onclick: move |_| {
                                                                let mut next = sizes_del.clone();
                                                                next.remove(idx);
                                                                on_change_spec.call(ProblemSpec::DisjointCycles { cycle_sizes: next });
                                                            },
                                                            "×"
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    if k < 8 {
                                        button {
                                            class: "btn-secondary",
                                            r#type: "button",
                                            disabled: running,
                                            onclick: move |_| {
                                                let mut next = sizes_add.clone();
                                                next.push(12);
                                                on_change_spec.call(ProblemSpec::DisjointCycles { cycle_sizes: next });
                                            },
                                            "+ Cycle"
                                        }
                                    }
                                }
                            }
                        }
                    }

                    ProblemSpec::BundledPreset { name } => rsx! {
                        div { class: "field",
                            label { "Sample Matrix" }
                            select {
                                disabled: running,
                                value: "{name}",
                                onchange: move |ev| {
                                    on_change_spec.call(ProblemSpec::BundledPreset { name: ev.value() });
                                },
                                for &(id, label_txt) in BUNDLED_PRESETS {
                                    option { value: "{id}", selected: id == name, "{label_txt}" }
                                }
                            }
                        }
                    },

                    ProblemSpec::RandomMatrix {
                        n,
                        family,
                        density,
                        log_spread,
                        ensure_matching,
                        seed,
                    } => {
                        let update = move |nn, nf, nd, nl, nem, ns| {
                            on_change_spec.call(ProblemSpec::RandomMatrix {
                                n: nn,
                                family: nf,
                                density: nd,
                                log_spread: nl,
                                ensure_matching: nem,
                                seed: ns,
                            });
                        };
                        rsx! {
                            div { class: "field",
                                label { "Matrix Family" }
                                div { class: "pill-row",
                                    for &(fam, fam_label) in RANDOM_FAMILIES {
                                        button {
                                            r#type: "button",
                                            disabled: running,
                                            class: if fam == family { "pill-choice active" } else { "pill-choice" },
                                            onclick: move |_| update(n, fam, density, log_spread, ensure_matching, seed),
                                            "{fam_label}"
                                        }
                                    }
                                }
                            }

                            div { class: "field-grid-2",
                                div { class: "field",
                                    label {
                                        span { "Dimension n" }
                                        span { class: "mono", "{n}×{n}" }
                                    }
                                    input {
                                        r#type: "number", min: "2", max: "256",
                                        disabled: running,
                                        value: "{n}",
                                        onchange: move |ev| {
                                            if let Ok(v) = ev.value().parse::<usize>() {
                                                update(v.clamp(2, 256), family, density, log_spread, ensure_matching, seed);
                                            }
                                        },
                                    }
                                }
                                div { class: "field",
                                    label {
                                        span { "Generator Seed" }
                                    }
                                    div { style: "display: flex; gap: 0.35rem;",
                                        input {
                                            r#type: "number", min: "0",
                                            disabled: running,
                                            value: "{seed}",
                                            onchange: move |ev| {
                                                if let Ok(v) = ev.value().parse::<u64>() {
                                                    update(n, family, density, log_spread, ensure_matching, v);
                                                }
                                            },
                                        }
                                        button {
                                            class: "btn-secondary",
                                            r#type: "button",
                                            disabled: running,
                                            onclick: move |_| update(n, family, density, log_spread, ensure_matching, next_seed(seed)),
                                            "New"
                                        }
                                    }
                                }
                            }

                            div { class: "field-grid-2",
                                div { class: "field",
                                    label {
                                        span { "Density" }
                                        span { class: "mono", "{(density * 100.0).round() as i32}%" }
                                    }
                                    input {
                                        r#type: "range", min: "0.05", max: "1.0", step: "0.05",
                                        disabled: running,
                                        value: "{density}",
                                        oninput: move |ev| {
                                            if let Ok(v) = ev.value().parse::<f64>() {
                                                update(n, family, v, log_spread, ensure_matching, seed);
                                            }
                                        },
                                    }
                                }
                                div { class: "field",
                                    label {
                                        span { "Log Spread" }
                                        span { class: "mono", "±{log_spread:.1}" }
                                    }
                                    input {
                                        r#type: "range", min: "0.0", max: "10.0", step: "0.5",
                                        disabled: running,
                                        value: "{log_spread}",
                                        oninput: move |ev| {
                                            if let Ok(v) = ev.value().parse::<f64>() {
                                                update(n, family, density, v, ensure_matching, seed);
                                            }
                                        },
                                    }
                                }
                            }

                            label { class: "toggle-item",
                                input {
                                    r#type: "checkbox",
                                    disabled: running,
                                    checked: ensure_matching,
                                    onchange: move |ev| {
                                        update(n, family, density, log_spread, ev.checked(), seed);
                                    },
                                }
                                "Plant guaranteed perfect matching"
                            }
                        }
                    }

                    ProblemSpec::MatrixMarket { text } => rsx! {
                        div { class: "field",
                            label {
                                span { "Matrix Market Coordinate" }
                                span { class: "pill-row",
                                    button {
                                        class: "pill-choice",
                                        r#type: "button",
                                        disabled: running,
                                        onclick: move |_| on_change_spec.call(ProblemSpec::MatrixMarket {
                                            text: SAMPLE_MTX_WEIGHTED.to_owned(),
                                        }),
                                        "4×4 weighted"
                                    }
                                    button {
                                        class: "pill-choice",
                                        r#type: "button",
                                        disabled: running,
                                        onclick: move |_| on_change_spec.call(ProblemSpec::MatrixMarket {
                                            text: SAMPLE_MTX_ZERO.to_owned(),
                                        }),
                                        "Zero permanent"
                                    }
                                }
                            }
                            textarea {
                                disabled: running,
                                value: "{text}",
                                onchange: move |ev| {
                                    on_change_spec.call(ProblemSpec::MatrixMarket { text: ev.value() });
                                },
                            }
                        }
                    },
                }
            }

            // ========================================================
            // Section 2 — Essential Run Configuration
            // ========================================================
            div { class: "control-group",
                h4 { class: "control-group-title",
                    span { "Run Configuration (WebGPU WGSL)" }
                }

                div { class: "field",
                    label {
                        span { "Target Relative Error (ε)" }
                        span { class: "mono", "±{(config.epsilon * 100.0):.1}%" }
                    }
                    div { class: "pill-row",
                        for eps in [0.25_f64, 0.1_f64, 0.05_f64] {
                            {
                                let cfg = config.clone();
                                let active = (config.epsilon - eps).abs() < 1e-9;
                                rsx! {
                                    button {
                                        r#type: "button",
                                        disabled: running,
                                        class: if active { "pill-choice active mono" } else { "pill-choice mono" },
                                        onclick: move |_| {
                                            let mut next = cfg.clone();
                                            next.epsilon = eps;
                                            on_change_config.call(next);
                                        },
                                        "ε = {eps}"
                                    }
                                }
                            }
                        }
                    }
                }

                div { class: "field-grid-3",
                    div { class: "field",
                        label { "Custom ε" }
                        {
                            let cfg = config.clone();
                            rsx! {
                                input {
                                    r#type: "number", min: "0.005", max: "1.0", step: "0.01",
                                    disabled: running,
                                    value: "{config.epsilon}",
                                    onchange: move |ev| {
                                        if let Ok(v) = ev.value().parse::<f64>() && v > 0.0 {
                                            let mut next = cfg.clone();
                                            next.epsilon = v.clamp(0.005, 1.0);
                                            on_change_config.call(next);
                                        }
                                    },
                                }
                            }
                        }
                    }
                    div { class: "field",
                        label { "Failure δ" }
                        {
                            let cfg = config.clone();
                            rsx! {
                                input {
                                    r#type: "number", min: "0.001", max: "0.5", step: "0.01",
                                    disabled: running,
                                    value: "{config.delta}",
                                    onchange: move |ev| {
                                        if let Ok(v) = ev.value().parse::<f64>() && v > 0.0 && v < 1.0 {
                                            let mut next = cfg.clone();
                                            next.delta = v.clamp(0.001, 0.5);
                                            on_change_config.call(next);
                                        }
                                    },
                                }
                            }
                        }
                    }
                    div { class: "field",
                        label { "Run Seed" }
                        {
                            let cfg = config.clone();
                            rsx! {
                                input {
                                    r#type: "number", min: "0",
                                    disabled: running,
                                    value: "{config.seed}",
                                    onchange: move |ev| {
                                        if let Ok(v) = ev.value().parse::<u64>() {
                                            let mut next = cfg.clone();
                                            next.seed = v;
                                            on_change_config.call(next);
                                        }
                                    },
                                }
                            }
                        }
                    }
                }

                div { class: "toggle-grid",
                    {
                        let cfg = config.clone();
                        rsx! {
                            label { class: "toggle-item",
                                input {
                                    r#type: "checkbox",
                                    disabled: running,
                                    checked: config.plan_only,
                                    onchange: move |ev| {
                                        let mut next = cfg.clone();
                                        next.plan_only = ev.checked();
                                        on_change_config.call(next);
                                    },
                                }
                                "Plan only (Phase 1)"
                            }
                        }
                    }
                    {
                        let cfg = config.clone();
                        rsx! {
                            label { class: "toggle-item",
                                input {
                                    r#type: "checkbox",
                                    disabled: running,
                                    checked: config.compute_exact,
                                    onchange: move |ev| {
                                        let mut next = cfg.clone();
                                        next.compute_exact = ev.checked();
                                        on_change_config.call(next);
                                    },
                                }
                                "Exact DP (n ≤ 20)"
                            }
                        }
                    }
                    {
                        let cfg = config.clone();
                        rsx! {
                            label { class: "toggle-item",
                                input {
                                    r#type: "checkbox",
                                    disabled: running,
                                    checked: config.capture_samples,
                                    onchange: move |ev| {
                                        let mut next = cfg.clone();
                                        next.capture_samples = ev.checked();
                                        on_change_config.call(next);
                                    },
                                }
                                "Stream live samples"
                            }
                        }
                    }
                    {
                        let cfg = config.clone();
                        rsx! {
                            label { class: "toggle-item",
                                input {
                                    r#type: "checkbox",
                                    disabled: running,
                                    checked: config.save_schedule,
                                    onchange: move |ev| {
                                        let mut next = cfg.clone();
                                        next.save_schedule = ev.checked();
                                        on_change_config.call(next);
                                    },
                                }
                                "Export schedule JSON"
                            }
                        }
                    }
                }
            }

            // ========================================================
            // Section 3 — Collapsible Advanced Engine Controls
            // ========================================================
            div { class: "accordion",
                button {
                    class: "accordion-trigger",
                    r#type: "button",
                    onclick: move |_| show_advanced.set(!show_advanced()),
                    span { "Advanced Algorithm & WebGPU Tuning" }
                    span { class: "mono", if show_advanced() { "−" } else { "+" } }
                }

                if show_advanced() {
                    div { class: "accordion-body",
                        div { class: "toggle-grid",
                            {
                                let cfg = config.clone();
                                rsx! {
                                    label { class: "toggle-item",
                                        input {
                                            r#type: "checkbox",
                                            disabled: running,
                                            checked: config.rigorous_schedule,
                                            onchange: move |ev| {
                                                let mut next = cfg.clone();
                                                next.rigorous_schedule = ev.checked();
                                                on_change_config.call(next);
                                            },
                                        }
                                        "Rigorous schedule"
                                    }
                                }
                            }
                            {
                                let cfg = config.clone();
                                rsx! {
                                    label { class: "toggle-item",
                                        input {
                                            r#type: "checkbox",
                                            disabled: running,
                                            checked: config.rigorous_sampler,
                                            onchange: move |ev| {
                                                let mut next = cfg.clone();
                                                next.rigorous_sampler = ev.checked();
                                                on_change_config.call(next);
                                            },
                                        }
                                        "Rigorous sampler"
                                    }
                                }
                            }
                        }

                        div { class: "field-grid-3",
                            div { class: "field",
                                label { "Move Time t" }
                                {
                                    let cfg = config.clone();
                                    rsx! {
                                        input {
                                            r#type: "number", min: "0.1", max: "20", step: "0.1",
                                            disabled: running,
                                            value: "{config.mutation_time}",
                                            onchange: move |ev| {
                                                if let Ok(v) = ev.value().parse::<f64>() && v > 0.0 {
                                                    let mut next = cfg.clone();
                                                    next.mutation_time = v;
                                                    on_change_config.call(next);
                                                }
                                            },
                                        }
                                    }
                                }
                            }
                            div { class: "field",
                                label { "Learn Move t" }
                                {
                                    let cfg = config.clone();
                                    rsx! {
                                        input {
                                            r#type: "number", min: "0", max: "20", step: "0.1",
                                            disabled: running,
                                            value: "{config.learn_mutation_time}",
                                            onchange: move |ev| {
                                                if let Ok(v) = ev.value().parse::<f64>() && v >= 0.0 {
                                                    let mut next = cfg.clone();
                                                    next.learn_mutation_time = v;
                                                    on_change_config.call(next);
                                                }
                                            },
                                        }
                                    }
                                }
                            }
                            div { class: "field",
                                label { "Density" }
                                {
                                    let cfg = config.clone();
                                    rsx! {
                                        input {
                                            r#type: "number", min: "1", max: "256", step: "1",
                                            disabled: running,
                                            value: "{config.density}",
                                            onchange: move |ev| {
                                                if let Ok(v) = ev.value().parse::<f64>() && v > 0.0 {
                                                    let mut next = cfg.clone();
                                                    next.density = v;
                                                    on_change_config.call(next);
                                                }
                                            },
                                        }
                                    }
                                }
                            }
                        }

                        div { class: "field-grid-2",
                            div { class: "field",
                                label { "Learn Particles (auto)" }
                                {
                                    let cfg = config.clone();
                                    rsx! {
                                        input {
                                            r#type: "number", min: "64", placeholder: "auto",
                                            disabled: running,
                                            value: "{config.learn_particles.map(|v| v.to_string()).unwrap_or_default()}",
                                            onchange: move |ev| {
                                                let mut next = cfg.clone();
                                                next.learn_particles = parse_opt_usize(&ev.value());
                                                on_change_config.call(next);
                                            },
                                        }
                                    }
                                }
                            }
                            div { class: "field",
                                label { "Particles / Island (auto)" }
                                {
                                    let cfg = config.clone();
                                    rsx! {
                                        input {
                                            r#type: "number", min: "16", placeholder: "auto",
                                            disabled: running,
                                            value: "{config.particles.map(|v| v.to_string()).unwrap_or_default()}",
                                            onchange: move |ev| {
                                                let mut next = cfg.clone();
                                                next.particles = parse_opt_usize(&ev.value());
                                                on_change_config.call(next);
                                            },
                                        }
                                    }
                                }
                            }
                            div { class: "field",
                                label { "Islands (auto)" }
                                {
                                    let cfg = config.clone();
                                    rsx! {
                                        input {
                                            r#type: "number", min: "4", placeholder: "auto",
                                            disabled: running,
                                            value: "{config.islands.map(|v| v.to_string()).unwrap_or_default()}",
                                            onchange: move |ev| {
                                                let mut next = cfg.clone();
                                                next.islands = parse_opt_usize(&ev.value());
                                                on_change_config.call(next);
                                            },
                                        }
                                    }
                                }
                            }
                            div { class: "field",
                                label { "Segments (auto)" }
                                {
                                    let cfg = config.clone();
                                    rsx! {
                                        input {
                                            r#type: "number", min: "1", max: "64", placeholder: "auto",
                                            disabled: running,
                                            value: "{config.segments.map(|v| v.to_string()).unwrap_or_default()}",
                                            onchange: move |ev| {
                                                let mut next = cfg.clone();
                                                next.segments = parse_opt_usize(&ev.value());
                                                on_change_config.call(next);
                                            },
                                        }
                                    }
                                }
                            }
                            div { class: "field",
                                label { "WGPU Working Set (MiB)" }
                                {
                                    let cfg = config.clone();
                                    rsx! {
                                        input {
                                            r#type: "number", min: "1", max: "512", placeholder: "32",
                                            disabled: running,
                                            value: "{config.wgpu_working_set_mb.map(|v| v.to_string()).unwrap_or_default()}",
                                            onchange: move |ev| {
                                                let mut next = cfg.clone();
                                                next.wgpu_working_set_mb = parse_opt_usize(&ev.value());
                                                on_change_config.call(next);
                                            },
                                        }
                                    }
                                }
                            }
                            div { class: "field",
                                label { "WGPU Event Budget" }
                                {
                                    let cfg = config.clone();
                                    rsx! {
                                        input {
                                            r#type: "number", min: "256", placeholder: "65536",
                                            disabled: running,
                                            value: "{config.wgpu_budget.map(|v| v.to_string()).unwrap_or_default()}",
                                            onchange: move |ev| {
                                                let mut next = cfg.clone();
                                                next.wgpu_budget = parse_opt_u32(&ev.value());
                                                on_change_config.call(next);
                                            },
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
