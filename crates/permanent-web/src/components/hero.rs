//! Full-viewport landing hero featuring the calm blue-yellow alternating-cycle
//! dimer animation and "Weighted Permanent Counting" typography.

use dioxus::prelude::*;

use crate::interop;
use crate::state::ResolvedTheme;

#[component]
pub fn Hero(resolved_theme: ResolvedTheme) -> Element {
    use_effect(move || {
        interop::hero_mount("hero-dimer-canvas", resolved_theme);
    });

    use_drop(|| {
        interop::hero_dispose();
    });

    rsx! {
        section { id: "top", class: "hero",
            div { class: "hero-stripe" }
            canvas {
                id: "hero-dimer-canvas",
                class: "hero-canvas",
                "aria-label": "Animated blue and yellow bipartite dimer tiling with alternating-cycle transitions",
            }
            div { class: "hero-scrim", "aria-hidden": "true" }

            div { class: "hero-content",
                div { class: "hero-copy",
                    div { class: "hero-eyebrow",
                        "WebGPU Population Annealing · Rust & WebAssembly"
                    }
                    h1 { class: "hero-title",
                        "Weighted Permanent Counting"
                    }
                    div { class: "hero-rule" }
                    p { class: "hero-subtitle",
                        "Estimate #P-complete matrix permanents and 2D/3D dimer partition functions directly on your GPU — with unbiased error bars, live schedule telemetry, and interactive covering exploration."
                    }

                    // Publication-grade display math equation with stacked limits
                    div {
                        class: "hero-formula",
                        role: "math",
                        "aria-label": "Permanent of A equals the sum over permutations sigma in S sub n of the product from u equals 1 to n of A sub u comma sigma of u",
                        span { class: "math-fn", "per" }
                        span { class: "math-punct", "(" }
                        span { class: "math-var", "A" }
                        span { class: "math-punct", ")" }
                        span { class: "math-eq", "=" }

                        span { class: "math-bigop",
                            span { class: "math-lim-top", "\u{00A0}" }
                            span { class: "math-sym", "∑" }
                            span { class: "math-lim-bot",
                                span { class: "math-var", "σ" }
                                span { " ∈ " }
                                span { class: "math-var", "S" }
                                sub { class: "math-var", "n" }
                            }
                        }

                        span { class: "math-bigop",
                            span { class: "math-lim-top",
                                span { class: "math-var", "n" }
                            }
                            span { class: "math-sym", "∏" }
                            span { class: "math-lim-bot",
                                span { class: "math-var", "u" }
                                span { " = 1" }
                            }
                        }

                        span { class: "math-term",
                            span { class: "math-var", "A" }
                            sub { class: "math-sub",
                                span { class: "math-var", "u" }
                                span { ", " }
                                span { class: "math-var", "σ" }
                                span { "(" }
                                span { class: "math-var", "u" }
                                span { ")" }
                            }
                        }
                    }

                    div { class: "hero-cta-row",
                        a { class: "btn-hero-primary", href: "#demos",
                            "Explore Interactive Demos"
                            span { "↓" }
                        }
                        a { class: "btn-hero-secondary", href: "#how-it-works",
                            "How It Works"
                        }
                    }
                }
            }

            a { class: "hero-hint", href: "#demos",
                span { "↓ Scroll to experiment · Click lattice to flip a cycle" }
            }
        }
    }
}
