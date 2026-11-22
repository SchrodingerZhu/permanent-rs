//! Concise three-card algorithmic summary and academic footer.

use dioxus::prelude::*;

#[component]
pub fn Footer() -> Element {
    rsx! {
        section { id: "how-it-works", class: "overview-section",
            div { class: "section-header",
                div {
                    h2 { class: "section-title", "How the Estimator Works" }
                    p { class: "section-sub",
                        "Unbiased sequential Monte Carlo over a learned inverse-temperature ladder, executing entirely in WGSL compute shaders."
                    }
                }
            }
            div { class: "overview-grid",
                article { class: "overview-card",
                    div { class: "overview-index", "01 · Combinatorics & Scaling" }
                    h3 { "Permanent = Weighted Perfect Matchings" }
                    p {
                        "Every nonnegative matrix defines a weighted bipartite graph; its permanent sums the weight products of all perfect matchings. Hungarian dual potentials balance rows and columns, and a small positive floor makes every permutation reachable before exact debiasing."
                    }
                }
                article { class: "overview-card",
                    div { class: "overview-index", "02 · Phase 1 & Phase 2" }
                    h3 { "Learned Hole Weights & Population Annealing" }
                    p {
                        "Phase 1 marches from β = 0 (uniform permutations) to β = 1 using the FAR chain to learn column hole weights and pick variance-bounded temperature steps. Phase 2 splits the schedule into independent segments, averages unbiased replica estimates, and checks segment boundaries."
                    }
                }
                article { class: "overview-card",
                    div { class: "overview-index", "03 · Browser Compute" }
                    h3 { "Native WGSL Shaders via WebGPU" }
                    p {
                        "Thousands of parallel chains advance simultaneously on your GPU using counter-based Philox streams and fixed-point energy accumulation. Computation runs asynchronously inside a Dedicated Web Worker so 3D inspection, slicing, and cancellation stay responsive."
                    }
                }
            }
        }

        footer { class: "site-footer",
            div { class: "footer-inner",
                span {
                    "permanent-rs · Built with Rust, Dioxus, WebAssembly, WebGPU (WGSL), and Three.js"
                }
                span {
                    "Algorithm based on relaxed-assignment Markov chains (Chen, Guo, Vigoda, Yang) · "
                    a {
                        href: "https://github.com/SchrodingerZhu/permanent-rs",
                        target: "_blank",
                        rel: "noreferrer",
                        "Source Repository"
                    }
                }
            }
        }
    }
}
