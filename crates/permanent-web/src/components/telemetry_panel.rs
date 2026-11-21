//! Purely typed live telemetry dashboard mirroring the terminal TUI using
//! structured KPI cards, confidence interval bar, three hairline SVG charts,
//! and collapsible diagnostic tables — zero formatted prose logs.

use dioxus::prelude::*;
use permanent_wasm::types::VerdictInfo;

use crate::interop;
use crate::state::{LevelPoint, RoundPoint, RunPhase, TelemetryState, format_compact_count, format_sig};

fn step_class(current: RunPhase, stage: usize) -> &'static str {
    let rank = match current {
        RunPhase::Idle => 0,
        RunPhase::Init => 1,
        RunPhase::Learn => 2,
        RunPhase::Plan => 3,
        RunPhase::Estimate => 4,
        RunPhase::Done => 6,
        RunPhase::Cancelled | RunPhase::Failed => 0,
    };
    if rank == 0 {
        "step-pill"
    } else if rank > stage {
        "step-pill done"
    } else if rank == stage {
        "step-pill active"
    } else {
        "step-pill"
    }
}

#[component]
pub fn TelemetryPanel(telemetry: TelemetryState) -> Element {
    let mut show_diag = use_signal(|| false);
    let mut copied_flash = use_signal(|| false);

    let phase = telemetry.current_phase();
    if phase == RunPhase::Idle && telemetry.summary.is_none() && telemetry.levels.is_empty() {
        return rsx! {
            div { class: "panel-scroll",
                div { class: "empty-telemetry",
                    strong { style: "display: block; margin-bottom: 0.35rem; color: var(--text-primary);",
                        "No active run yet"
                    }
                    "Click "
                    strong { "Run on WebGPU" }
                    " above to stream live Phase 1 hole weights, temperature schedule steps, unbiased population samples, and Phase 2 convergence."
                }
            }
        };
    }

    let sweep_pct = telemetry
        .sweep
        .as_ref()
        .map(|s| if s.total > 0 { ((s.done as f64 / s.total as f64) * 100.0).clamp(0.0, 100.0) } else { 0.0 })
        .unwrap_or(0.0);

    let last_round = telemetry.rounds.last().cloned();
    let final_est = telemetry.summary.as_ref().and_then(|s| s.estimate.as_ref());

    rsx! {
        div { class: "panel-scroll",
            // 1. 5-Stage Pipeline Stepper
            div { class: "stepper",
                div { class: step_class(phase, 1), "1 Scale" }
                div { class: step_class(phase, 2), "2 Learn" }
                div { class: step_class(phase, 3), "3 Plan" }
                div { class: step_class(phase, 4), "4 Estimate" }
                div { class: step_class(phase, 5), "5 Done" }
            }

            // Status notices (cancelled / failed / zero permanent / plan-only)
            if phase == RunPhase::Cancelled {
                div { class: "notice-banner warn",
                    "Run cancelled cooperatively at the last checkpoint."
                }
            }
            if let Some(ref err) = telemetry.error {
                div { class: "notice-banner err", "{err}" }
            }
            if let Some(ref sum) = telemetry.summary && sum.zero_permanent {
                div { class: "result-card",
                    div { class: "result-main",
                        span { class: "result-value", "per(A) = 0" }
                        span { class: "chip warn", "No perfect matching" }
                    }
                    span { style: "font-size: 0.82rem; color: var(--text-secondary);",
                        "Hungarian scaling detected no finite-weight perfect matching on the matrix support."
                    }
                }
            }

            // Live Phase 2 Sweep Progress Bar
            if let Some(ref sw) = telemetry.sweep && telemetry.is_running() {
                div { class: "progress-wrap",
                    div { class: "progress-head",
                        span { class: "mono",
                            "Phase 2 · Round {sw.round + 1} · Segment {sw.segment + 1}"
                        }
                        span { class: "mono",
                            "{sw.done} / {sw.total} sweeps ({sweep_pct:.0}%)"
                        }
                    }
                    div { class: "progress-track",
                        div { class: "progress-fill", style: "width: {sweep_pct:.1}%;" }
                    }
                }
            }

            // 2. Primary Estimate Card (Final EstimateSummary or Live Round)
            if let Some(est) = final_est {
                {
                    let exact_ln = telemetry.exact_ln;
                    let half_span = (est.ln_ci_high - est.ln_ci_low).max(1e-9);
                    let mut min_x = est.ln_ci_low - half_span * 0.35;
                    let mut max_x = est.ln_ci_high + half_span * 0.35;
                    if let Some(ex) = exact_ln {
                        min_x = min_x.min(ex - half_span * 0.15);
                        max_x = max_x.max(ex + half_span * 0.15);
                    }
                    let span_x = (max_x - min_x).max(1e-9);
                    let pct = |x: f64| -> f64 { (((x - min_x) / span_x) * 100.0).clamp(1.0, 99.0) };
                    let ci_left = pct(est.ln_ci_low);
                    let ci_width = (pct(est.ln_ci_high) - ci_left).max(1.5);
                    let pt_left = pct(est.ln_permanent);

                    let verdict_chip = match est.verdict {
                        VerdictInfo::Consistent { max_gap_se } => ("chip ok mono", format!("Boundary ok ({max_gap_se:+.2}σ)")),
                        VerdictInfo::Inconsistent { max_gap_se } => ("chip err mono", format!("Boundary gap {max_gap_se:+.2}σ")),
                        VerdictInfo::Unchecked => ("chip", "Single segment".to_owned()),
                    };

                    let exact_chip = est.rel_error_pct.map(|rel| {
                        let cls = match est.z_score.map(f64::abs) {
                            Some(z) if z <= 2.0 => "chip ok mono",
                            Some(z) if z <= 3.5 => "chip warn mono",
                            Some(_) => "chip err mono",
                            None => "chip exact mono",
                        };
                        let label = match est.z_score {
                            Some(z) => format!("vs exact: {rel:+.2}% ({z:+.2}σ)"),
                            None => format!("vs exact: {rel:+.2}%"),
                        };
                        (cls, label)
                    });

                    rsx! {
                        div { class: "result-card",
                            div { class: "result-main",
                                span { class: "result-value", "per ≈ {est.scientific}" }
                                span { class: "result-ln", "ln = {est.ln_permanent:.5} ± {est.std_error:.5}" }
                            }

                            div { class: "ci-box",
                                div { class: "ci-labels mono",
                                    span { "{est.ci_level_pct:.1}% CI: [{est.ci_low_scientific}, {est.ci_high_scientific}]" }
                                    span { "±{est.rel_half_width_pct:.2}%" }
                                }
                                div { class: "ci-track",
                                    div {
                                        class: "ci-interval",
                                        style: "left: {ci_left:.1}%; width: {ci_width:.1}%;",
                                    }
                                    div {
                                        class: "ci-point",
                                        style: "left: {pt_left:.1}%;",
                                    }
                                    if let Some(ex) = exact_ln {
                                        div {
                                            class: "ci-exact-tick",
                                            title: "Exact ln(per) = {ex:.6}",
                                            style: "left: {pct(ex):.1}%;",
                                        }
                                    }
                                }
                            }

                            if let (Some(s), Some(se)) = (est.entropy_per_site, est.entropy_std_error) {
                                div { style: "display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0.4rem; font-size: 0.8rem;",
                                    span { class: "mono", "Entropy / site = {s:.5} ± {se:.5}" }
                                    if let Some(lb) = est.entropy_lower_bound {
                                        span { class: "chip exact mono", "Schrijver ≥ {lb:.4}" }
                                    }
                                }
                            }

                            div { class: "stage-chips",
                                span { class: "{verdict_chip.0}", "{verdict_chip.1}" }
                                if let Some((cls, txt)) = exact_chip {
                                    span { class: "{cls}", "{txt}" }
                                }
                                if !est.met_target {
                                    span { class: "chip warn", "Target error not reached" }
                                }
                                if est.mutate_over_budget + est.sampler_over_budget > 0 {
                                    span { class: "chip warn mono",
                                        "{est.mutate_over_budget} mut / {est.sampler_over_budget} smp capped"
                                    }
                                }
                                if telemetry.exact_skipped_too_large {
                                    span { class: "chip", "Subset DP skipped (n > 20)" }
                                }
                                span { class: "chip mono", "{est.seconds:.2}s · {est.islands} islands" }
                            }
                        }
                    }
                }
            } else if let Some(ref rd) = last_round {
                div { class: "result-card",
                    div { class: "result-main",
                        span { class: "result-value", "ln = {rd.ln_permanent:.5} ± {rd.std_error:.5}" }
                        span { class: "chip live mono", "Round {rd.round + 1} · {rd.islands} islands" }
                    }
                    div { class: "stage-chips",
                        span { class: "chip mono", "{format_compact_count(rd.mutate_events)} mut" }
                        span { class: "chip mono", "{format_compact_count(rd.sampler_events)} smp" }
                        span { class: "chip mono", "{rd.seconds:.1}s" }
                    }
                }
            } else if let Some(ref sum) = telemetry.summary && sum.plan_only && !sum.zero_permanent {
                div { class: "result-card",
                    div { class: "result-main",
                        span { class: "result-value", "Plan Ready" }
                        span { class: "chip ok", "Plan-only mode" }
                    }
                    if let Some(ref p) = sum.plan {
                        span { class: "mono", style: "font-size: 0.82rem; color: var(--text-secondary);",
                            "{p.segments.len()} segments · {p.islands} islands × {p.particles} particles · target σ = {p.sigma_target:.4}"
                        }
                    }
                }
            }

            // 3. Chart A — Hole Weights w(v) vs Temperature β
            if !telemetry.levels.is_empty() {
                HoleWeightsChart { levels: telemetry.levels.clone() }
            }

            // 4. Chart B — Schedule Step Relvar & Phase 2 Segment Lanes
            if !telemetry.levels.is_empty() || telemetry.plan.is_some() {
                ScheduleAndSegmentsChart {
                    levels: telemetry.levels.clone(),
                    segments: telemetry.plan.as_ref().map(|p| p.segments.clone()).unwrap_or_default(),
                    segment_done: telemetry.sweep.as_ref().map(|s| s.segment_done.clone()).unwrap_or_default(),
                }
            }

            // 5. Chart C — Round Convergence
            if !telemetry.rounds.is_empty() {
                RoundConvergenceChart {
                    rounds: telemetry.rounds.clone(),
                    exact_ln: telemetry.exact_ln,
                    z: telemetry.plan.as_ref().map(|p| p.z).unwrap_or(1.96),
                }
            }

            // 6. Collapsible Structured Diagnostics Tables & JSON Export
            if telemetry.phase1.is_some() || telemetry.plan.is_some() || telemetry.summary.is_some() {
                div { class: "accordion",
                    button {
                        class: "accordion-trigger",
                        r#type: "button",
                        onclick: move |_| show_diag.set(!show_diag()),
                        span { "Structured Diagnostics" }
                        span { class: "mono", if show_diag() { "−" } else { "+" } }
                    }
                    if show_diag() {
                        div { class: "accordion-body",
                            if let Some(ref p1) = telemetry.phase1 {
                                h4 { class: "control-group-title", "Phase 1 · Schedule Learning" }
                                table { class: "diag-table",
                                    tbody {
                                        tr { th { "Temperatures" } td { "{p1.temperatures}" } }
                                        tr { th { "Duration" } td { "{p1.seconds:.2} s" } }
                                        tr { th { "Population" } td { "{p1.learn_particles} particles" } }
                                        tr { th { "Learning passes" } td { "{p1.learn_iterations}" } }
                                        tr { th { "FAR steps" } td { "{format_compact_count(p1.far_steps)}" } }
                                        if p1.mutate_events > 0 {
                                            tr { th { "Phase 1 mutations" } td { "{format_compact_count(p1.mutate_events)}" } }
                                        }
                                        tr { th { "Worst 1st correction" } td { "×{p1.worst_first_correction_factor:.3}" } }
                                        tr { th { "Weight check (~1)" } td { "{p1.check_min:.3} .. {p1.check_max:.3}" } }
                                        if p1.learn_capped_chains + p1.mutate_over_budget > 0 {
                                            tr { th { "Capped FAR / mut" } td { "{p1.learn_capped_chains} / {p1.mutate_over_budget}" } }
                                        }
                                    }
                                }
                            }

                            if let Some(ref pl) = telemetry.plan {
                                h4 { class: "control-group-title", "Plan · Phase 2 Sizing" }
                                table { class: "diag-table",
                                    tbody {
                                        tr { th { "Segments" } td { "{pl.segments.len()}" } }
                                        tr { th { "Islands × particles" } td { "{pl.islands} × {pl.particles}" } }
                                        tr { th { "Planned island relvar" } td { "{pl.island_relvar:.4} (Σ={pl.relvar_sum:.3})" } }
                                        tr { th { "Mixing bound factor" } td { "{pl.mixing_factor:.2}" } }
                                        tr { th { "Target σ / z" } td { "{pl.sigma_target:.4} / {pl.z:.3}" } }
                                        tr { th { "Sampler stage time" } td { "{pl.stage_time:.4}" } }
                                        if pl.predicted_events > 0.0 {
                                            tr { th { "Predicted events" } td { "{pl.predicted_events:.2e}" } }
                                        }
                                    }
                                }
                            }

                            if let Some(est) = final_est {
                                h4 { class: "control-group-title", "Phase 2 · Estimation Totals" }
                                table { class: "diag-table",
                                    tbody {
                                        tr { th { "Measured vs planned var" } td { "{est.island_relvar:.4} vs {est.planned_island_relvar:.4}" } }
                                        tr { th { "Regularization E[g]" } td { "{est.mean_gap:.4}" } }
                                        tr { th { "Rounds" } td { "{est.rounds}" } }
                                        tr { th { "Mutation events" } td { "{format_compact_count(est.mutate_events)}" } }
                                        tr { th { "Sampler events" } td { "{format_compact_count(est.sampler_events)}" } }
                                        tr { th { "Phase 2 duration" } td { "{est.seconds:.2} s" } }
                                    }
                                }
                            }

                            if let Some(ref summary) = telemetry.summary {
                                div { class: "pill-row",
                                    {
                                        let sum_copy = summary.clone();
                                        rsx! {
                                            button {
                                                class: "btn-secondary",
                                                r#type: "button",
                                                onclick: move |_| {
                                                    if let Ok(json_txt) = serde_json::to_string_pretty(&sum_copy) {
                                                        interop::copy_to_clipboard(&json_txt);
                                                        copied_flash.set(true);
                                                    }
                                                },
                                                if copied_flash() { "Copied Summary JSON ✓" } else { "Copy Summary JSON" }
                                            }
                                        }
                                    }
                                    if let Some(ref sched) = summary.schedule_json {
                                        {
                                            let sched_clone = sched.clone();
                                            let n_val = summary.n;
                                            rsx! {
                                                button {
                                                    class: "btn-secondary",
                                                    r#type: "button",
                                                    onclick: move |_| {
                                                        interop::download_json(&format!("schedule-n{n_val}.json"), &sched_clone);
                                                    },
                                                    "Download Schedule JSON"
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
    }
}

#[component]
fn HoleWeightsChart(levels: Vec<LevelPoint>) -> Element {
    let w = 440.0_f64;
    let h = 152.0_f64;
    let pad_l = 42.0_f64;
    let pad_r = 14.0_f64;
    let pad_t = 12.0_f64;
    let pad_b = 24.0_f64;
    let plot_w = w - pad_l - pad_r;
    let plot_h = h - pad_t - pad_b;

    let n_cols = levels.first().map(|l| l.log_w.len()).unwrap_or(0);
    if n_cols == 0 {
        return rsx! {};
    }

    let mut min_lw = f64::INFINITY;
    let mut max_lw = f64::NEG_INFINITY;
    for lv in &levels {
        for &lw in &lv.log_w {
            if lw.is_finite() {
                min_lw = min_lw.min(lw);
                max_lw = max_lw.max(lw);
            }
        }
    }
    if !min_lw.is_finite() || !max_lw.is_finite() {
        return rsx! {};
    }
    let span = (max_lw - min_lw).max(0.5);
    let min_y = min_lw - span * 0.08;
    let max_y = max_lw + span * 0.08;
    let y_span = (max_y - min_y).max(1e-6);

    let x_of = |beta: f64| pad_l + beta.clamp(0.0, 1.0) * plot_w;
    let y_of = |lw: f64| pad_t + plot_h - ((lw - min_y) / y_span) * plot_h;

    // Pick lowest and highest final column curves to highlight cleanly in Blue & Gold.
    let last_lw = &levels.last().unwrap().log_w;
    let (mut min_col, mut max_col) = (0usize, 0usize);
    for (c, &v) in last_lw.iter().enumerate() {
        if v < last_lw[min_col] {
            min_col = c;
        }
        if v > last_lw[max_col] {
            max_col = c;
        }
    }

    // Cap background curves cleanly to 32 sampled columns if n is huge.
    let stride = (n_cols / 32).max(1);
    let mut bg_paths: Vec<String> = Vec::new();
    for c in (0..n_cols).step_by(stride) {
        if c == min_col || c == max_col {
            continue;
        }
        let mut pts = String::new();
        for lv in &levels {
            if let Some(&lw) = lv.log_w.get(c)
                && lw.is_finite()
            {
                if !pts.is_empty() {
                    pts.push(' ');
                }
                pts.push_str(&format!("{:.1},{:.1}", x_of(lv.beta), y_of(lw)));
            }
        }
        if !pts.is_empty() {
            bg_paths.push(pts);
        }
    }

    let build_path = |col: usize| -> String {
        let mut pts = String::new();
        for lv in &levels {
            if let Some(&lw) = lv.log_w.get(col)
                && lw.is_finite()
            {
                if !pts.is_empty() {
                    pts.push(' ');
                }
                pts.push_str(&format!("{:.1},{:.1}", x_of(lv.beta), y_of(lw)));
            }
        }
        pts
    };

    let min_path = build_path(min_col);
    let max_path = build_path(max_col);
    let top_val = max_lw.exp();
    let bot_val = min_lw.exp();

    rsx! {
        div { class: "chart-card",
            div { class: "chart-title-row",
                h4 { class: "chart-title", "Hole Weights ln w(v) vs β" }
                span { class: "chart-caption mono", "{levels.len()} temperatures" }
            }
            svg { class: "chart-svg", view_box: "0 0 {w} {h}",
                // Horizontal reference gridlines
                for frac in [0.0_f64, 0.5, 1.0] {
                    line {
                        x1: "{pad_l}",
                        y1: "{pad_t + frac * plot_h:.1}",
                        x2: "{w - pad_r}",
                        y2: "{pad_t + frac * plot_h:.1}",
                        stroke: "var(--chart-grid)",
                        stroke_width: "1",
                    }
                }
                // Axes
                line { x1: "{pad_l}", y1: "{pad_t}", x2: "{pad_l}", y2: "{pad_t + plot_h}", stroke: "var(--border-strong)", stroke_width: "1" }
                line { x1: "{pad_l}", y1: "{pad_t + plot_h}", x2: "{w - pad_r}", y2: "{pad_t + plot_h}", stroke: "var(--border-strong)", stroke_width: "1" }

                // Muted column trajectories
                for (i, pts) in bg_paths.iter().enumerate() {
                    polyline {
                        key: "hw-{i}",
                        points: "{pts}",
                        fill: "none",
                        stroke: "var(--chart-muted)",
                        stroke_width: "1",
                        stroke_opacity: "0.7",
                    }
                }

                // Highlighted extreme curves
                if !max_path.is_empty() {
                    polyline {
                        points: "{max_path}",
                        fill: "none",
                        stroke: "var(--dimer-z)",
                        stroke_width: "2.2",
                    }
                }
                if !min_path.is_empty() {
                    polyline {
                        points: "{min_path}",
                        fill: "none",
                        stroke: "var(--dimer-x)",
                        stroke_width: "2.2",
                    }
                }

                // Labels
                text { x: "{pad_l - 6.0}", y: "{pad_t + 7.0}", text_anchor: "end", fill: "var(--text-muted)", font_size: "9.5", class: "mono", "{format_sig(top_val)}" }
                text { x: "{pad_l - 6.0}", y: "{pad_t + plot_h}", text_anchor: "end", fill: "var(--text-muted)", font_size: "9.5", class: "mono", "{format_sig(bot_val)}" }
                text { x: "{pad_l}", y: "{h - 6.0}", fill: "var(--text-muted)", font_size: "9.5", class: "mono", "β=0" }
                text { x: "{pad_l + plot_w * 0.5}", y: "{h - 6.0}", text_anchor: "middle", fill: "var(--text-muted)", font_size: "9.5", class: "mono", "0.5" }
                text { x: "{w - pad_r}", y: "{h - 6.0}", text_anchor: "end", fill: "var(--text-muted)", font_size: "9.5", class: "mono", "β=1" }
            }
        }
    }
}

#[component]
fn ScheduleAndSegmentsChart(levels: Vec<LevelPoint>, segments: Vec<[usize; 2]>, segment_done: Vec<usize>) -> Element {
    let w = 440.0_f64;
    let seg_h = if segments.is_empty() { 0.0 } else { 22.0 };
    let h = 108.0 + seg_h;
    let pad_l = 42.0_f64;
    let pad_r = 14.0_f64;
    let pad_t = 12.0_f64;
    let plot_w = w - pad_l - pad_r;
    let bar_h = 68.0_f64;

    let count = levels.len().max(1);
    let max_rv = levels.iter().map(|l| l.relvar).fold(0.32_f64, f64::max);
    let bar_slot = plot_w / count as f64;
    let target_y = pad_t + bar_h - (0.25 / max_rv).clamp(0.0, 1.0) * bar_h;

    rsx! {
        div { class: "chart-card",
            div { class: "chart-title-row",
                h4 { class: "chart-title", "Step Relative Variance & Segments" }
                span { class: "chart-caption mono",
                    if !segments.is_empty() { "{segments.len()} segments · target ≤ 0.25" } else { "target ≤ 0.25" }
                }
            }
            svg { class: "chart-svg", view_box: "0 0 {w} {h}",
                // 0.25 dashed target budget line
                line {
                    x1: "{pad_l}",
                    y1: "{target_y:.1}",
                    x2: "{w - pad_r}",
                    y2: "{target_y:.1}",
                    stroke: "var(--dimer-z)",
                    stroke_width: "1.2",
                    stroke_dasharray: "4 3",
                }
                line { x1: "{pad_l}", y1: "{pad_t + bar_h}", x2: "{w - pad_r}", y2: "{pad_t + bar_h}", stroke: "var(--border-strong)", stroke_width: "1" }
                text { x: "{pad_l - 6.0}", y: "{target_y + 3.0:.1}", text_anchor: "end", fill: "var(--accent-gold-text)", font_size: "9", class: "mono", "0.25" }
                text { x: "{pad_l - 6.0}", y: "{pad_t + bar_h}", text_anchor: "end", fill: "var(--text-muted)", font_size: "9", class: "mono", "0" }

                for (i, lv) in levels.iter().enumerate() {
                    {
                        let bh = ((lv.relvar / max_rv).clamp(0.0, 1.0) * bar_h).max(1.0);
                        let bx = pad_l + i as f64 * bar_slot;
                        let bw = (bar_slot - 1.2).max(1.5);
                        rsx! {
                            rect {
                                key: "rv-{i}",
                                x: "{bx:.1}",
                                y: "{pad_t + bar_h - bh:.1}",
                                width: "{bw:.1}",
                                height: "{bh:.1}",
                                rx: "1.5",
                                fill: if lv.relvar > 0.28 { "var(--dimer-z)" } else { "var(--dimer-x)" },
                                fill_opacity: "0.85",
                            }
                        }
                    }
                }

                // Segment partition lanes underneath
                if !segments.is_empty() {
                    {
                        let lane_y = pad_t + bar_h + 10.0;
                        let total_steps = segments.last().map(|s| s[1]).unwrap_or(count).max(1) as f64;
                        rsx! {
                            for (si, &[start, end]) in segments.iter().enumerate() {
                                {
                                    let x0 = pad_l + (start as f64 / total_steps) * plot_w;
                                    let x1 = pad_l + (end as f64 / total_steps) * plot_w;
                                    let sw = (x1 - x0 - 3.0).max(4.0);
                                    let done_lvl = segment_done.get(si).copied().unwrap_or(0);
                                    let frac = if end > start {
                                        ((done_lvl.saturating_sub(start)) as f64 / (end - start) as f64).clamp(0.0, 1.0)
                                    } else {
                                        1.0
                                    };
                                    let fill_w = sw * frac;
                                    rsx! {
                                        g { key: "seg-{si}",
                                            rect {
                                                x: "{x0:.1}",
                                                y: "{lane_y:.1}",
                                                width: "{sw:.1}",
                                                height: "11",
                                                rx: "3",
                                                fill: "var(--bg-subtle)",
                                                stroke: "var(--border-strong)",
                                                stroke_width: "1",
                                            }
                                            if fill_w > 0.5 {
                                                rect {
                                                    x: "{x0:.1}",
                                                    y: "{lane_y:.1}",
                                                    width: "{fill_w:.1}",
                                                    height: "11",
                                                    rx: "3",
                                                    fill: if si % 2 == 0 { "var(--dimer-x)" } else { "var(--dimer-z)" },
                                                    fill_opacity: "0.8",
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
    }
}

#[component]
fn RoundConvergenceChart(rounds: Vec<RoundPoint>, exact_ln: Option<f64>, z: f64) -> Element {
    let w = 440.0_f64;
    let h = 136.0_f64;
    let pad_l = 52.0_f64;
    let pad_r = 16.0_f64;
    let pad_t = 12.0_f64;
    let pad_b = 24.0_f64;
    let plot_w = w - pad_l - pad_r;
    let plot_h = h - pad_t - pad_b;

    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for r in &rounds {
        let half = z * r.std_error;
        min_y = min_y.min(r.ln_permanent - half);
        max_y = max_y.max(r.ln_permanent + half);
    }
    if let Some(ex) = exact_ln {
        min_y = min_y.min(ex);
        max_y = max_y.max(ex);
    }
    let span = (max_y - min_y).max(0.02);
    min_y -= span * 0.14;
    max_y += span * 0.14;
    let y_span = (max_y - min_y).max(1e-6);

    let y_of = |v: f64| pad_t + plot_h - ((v - min_y) / y_span) * plot_h;
    let count = rounds.len();
    let x_of = |idx: usize| {
        if count <= 1 { pad_l + plot_w * 0.5 } else { pad_l + (idx as f64 / (count - 1) as f64) * plot_w }
    };

    rsx! {
        div { class: "chart-card",
            div { class: "chart-title-row",
                h4 { class: "chart-title", "Phase 2 Round Convergence (ln per ± zσ)" }
                span { class: "chart-caption mono", "{count} round(s)" }
            }
            svg { class: "chart-svg", view_box: "0 0 {w} {h}",
                line { x1: "{pad_l}", y1: "{pad_t}", x2: "{pad_l}", y2: "{pad_t + plot_h}", stroke: "var(--border-strong)", stroke_width: "1" }
                line { x1: "{pad_l}", y1: "{pad_t + plot_h}", x2: "{w - pad_r}", y2: "{pad_t + plot_h}", stroke: "var(--border-strong)", stroke_width: "1" }

                if let Some(ex) = exact_ln {
                    line {
                        x1: "{pad_l}",
                        y1: "{y_of(ex):.1}",
                        x2: "{w - pad_r}",
                        y2: "{y_of(ex):.1}",
                        stroke: "var(--dimer-z)",
                        stroke_width: "1.6",
                        stroke_dasharray: "5 3",
                    }
                }

                for (i, r) in rounds.iter().enumerate() {
                    {
                        let cx = x_of(i);
                        let cy = y_of(r.ln_permanent);
                        let y_hi = y_of(r.ln_permanent + z * r.std_error);
                        let y_lo = y_of(r.ln_permanent - z * r.std_error);
                        rsx! {
                            g { key: "rd-{i}",
                                line { x1: "{cx:.1}", y1: "{y_hi:.1}", x2: "{cx:.1}", y2: "{y_lo:.1}", stroke: "var(--dimer-x)", stroke_width: "2" }
                                line { x1: "{cx - 4.0:.1}", y1: "{y_hi:.1}", x2: "{cx + 4.0:.1}", y2: "{y_hi:.1}", stroke: "var(--dimer-x)", stroke_width: "1.6" }
                                line { x1: "{cx - 4.0:.1}", y1: "{y_lo:.1}", x2: "{cx + 4.0:.1}", y2: "{y_lo:.1}", stroke: "var(--dimer-x)", stroke_width: "1.6" }
                                circle { cx: "{cx:.1}", cy: "{cy:.1}", r: "3.6", fill: "var(--accent-primary)" }
                                text {
                                    x: "{cx:.1}",
                                    y: "{h - 6.0}",
                                    text_anchor: "middle",
                                    fill: "var(--text-muted)",
                                    font_size: "9",
                                    class: "mono",
                                    "R{r.round + 1}"
                                }
                            }
                        }
                    }
                }

                text { x: "{pad_l - 6.0}", y: "{pad_t + 8.0}", text_anchor: "end", fill: "var(--text-muted)", font_size: "9", class: "mono", "{max_y:.2}" }
                text { x: "{pad_l - 6.0}", y: "{pad_t + plot_h}", text_anchor: "end", fill: "var(--text-muted)", font_size: "9", class: "mono", "{min_y:.2}" }
            }
        }
    }
}
