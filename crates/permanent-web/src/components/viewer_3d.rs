//! Three.js 3D cubic lattice viewport component with minimal architectural HUD.

use dioxus::prelude::*;
use permanent_demo::types::ProblemGeometry;

use crate::interop;
use crate::state::ResolvedTheme;

#[component]
pub fn Viewer3d(
    geometry: ProblemGeometry,
    matching: Option<Vec<u32>>,
    valid: bool,
    resolved_theme: ResolvedTheme,
) -> Element {
    let mut projection = use_signal(|| "ortho");
    let mut slice_z = use_signal(|| None::<usize>);
    let mut show_periodic = use_signal(|| true);

    let (dims, periodic) = match &geometry {
        ProblemGeometry::Lattice3d { dims, periodic, .. } => (*dims, *periodic),
        _ => ([4, 4, 4], false),
    };

    // Keep slice index valid when lattice dimensions change.
    use_effect(use_reactive((&dims,), move |(d,)| {
        let current = *slice_z.peek();
        if let Some(z) = current
            && z >= d[2]
        {
            slice_z.set(None);
            interop::three_set_slice_z(None);
        }
    }));

    // Mount Three.js canvas and rebuild lattice geometry on problem/theme change.
    let initial_matching = matching.clone();
    use_effect(use_reactive((&geometry, &resolved_theme), move |(geo, theme)| {
        interop::three_mount("three-lattice-canvas", theme);
        interop::three_set_problem(&geo, initial_matching.as_deref(), valid);
    }));

    // Fast InstancedMesh-only update when only the active covering changes.
    use_effect(use_reactive((&matching, &valid), move |(m, v)| {
        interop::three_set_matching(m.as_deref(), v);
    }));

    use_drop(|| {
        interop::three_dispose();
    });

    let lz = dims[2];
    let current_slice = slice_z();
    let slice_label = match current_slice {
        None => "All".to_owned(),
        Some(z) => format!("z={z}"),
    };

    rsx! {
        div { class: "viewport-body",
            canvas {
                id: "three-lattice-canvas",
                class: "three-canvas",
                "aria-label": "Interactive 3D cubic lattice dimer model. Drag to orbit, scroll to zoom.",
            }

            div { class: "hud-bar",
                button {
                    class: "hud-btn",
                    r#type: "button",
                    onclick: move |_| interop::three_set_view_preset("iso"),
                    "Iso"
                }
                button {
                    class: "hud-btn",
                    r#type: "button",
                    onclick: move |_| interop::three_set_view_preset("front"),
                    "Front"
                }
                button {
                    class: "hud-btn",
                    r#type: "button",
                    onclick: move |_| interop::three_set_view_preset("top"),
                    "Top"
                }

                span { class: "hud-sep" }

                button {
                    class: if projection() == "ortho" { "hud-btn active" } else { "hud-btn" },
                    r#type: "button",
                    onclick: move |_| {
                        projection.set("ortho");
                        interop::three_set_projection("ortho");
                    },
                    "Ortho"
                }
                button {
                    class: if projection() == "persp" { "hud-btn active" } else { "hud-btn" },
                    r#type: "button",
                    onclick: move |_| {
                        projection.set("persp");
                        interop::three_set_projection("persp");
                    },
                    "Persp"
                }

                span { class: "hud-sep" }

                label { class: "hud-slice",
                    span { class: "mono", "Slice: {slice_label}" }
                    input {
                        r#type: "range",
                        min: "-1",
                        max: "{lz as isize - 1}",
                        step: "1",
                        value: "{current_slice.map(|z| z as isize).unwrap_or(-1)}",
                        oninput: move |ev| {
                            if let Ok(v) = ev.value().parse::<isize>() {
                                let next = if v < 0 { None } else { Some(v as usize) };
                                slice_z.set(next);
                                interop::three_set_slice_z(next);
                            }
                        },
                    }
                }

                if periodic {
                    span { class: "hud-sep" }
                    button {
                        class: if show_periodic() { "hud-btn active" } else { "hud-btn" },
                        r#type: "button",
                        onclick: move |_| {
                            let next = !show_periodic();
                            show_periodic.set(next);
                            interop::three_set_show_periodic(next);
                        },
                        "Wraps"
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
                    "X"
                }
                span { class: "legend-item",
                    span { class: "swatch-rod", style: "background: var(--dimer-y);" }
                    "Y"
                }
                span { class: "legend-item",
                    span { class: "swatch-rod", style: "background: var(--dimer-z);" }
                    "Z"
                }
            }
        }
    }
}
