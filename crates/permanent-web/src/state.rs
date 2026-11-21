//! Reactive UI and telemetry state models for the Dioxus frontend.

use permanent_wasm::types::{Capabilities, Phase1Summary, PlanSummary, RunSummary, SampleSnapshot, WebEvent};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeMode {
    System,
    Light,
    Dark,
}

impl ThemeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Light => "light",
            Self::Dark => "dark",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "light" => Self::Light,
            "dark" => Self::Dark,
            _ => Self::System,
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::System => Self::Light,
            Self::Light => Self::Dark,
            Self::Dark => Self::System,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::System => "Auto",
            Self::Light => "Light",
            Self::Dark => "Dark",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedTheme {
    Light,
    Dark,
}

impl ResolvedTheme {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Light => "light",
            Self::Dark => "dark",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveTab {
    Configure,
    Telemetry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Viewer2dMode {
    Bonds,
    Dominoes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunPhase {
    Idle,
    Init,
    Learn,
    Plan,
    Estimate,
    Done,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LevelPoint {
    pub index: usize,
    pub beta: f64,
    pub step: f64,
    pub relvar: f64,
    pub log_w: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoundPoint {
    pub round: usize,
    pub islands: usize,
    pub ln_permanent: f64,
    pub std_error: f64,
    pub mutate_events: u64,
    pub sampler_events: u64,
    pub seconds: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SweepProgress {
    pub round: usize,
    pub done: u64,
    pub total: u64,
    pub segment: usize,
    pub level: usize,
    pub islands: usize,
    pub segment_done: Vec<usize>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TelemetryState {
    pub phase: Option<RunPhase>,
    pub engine: Option<String>,
    pub n: usize,
    pub nnz: usize,
    pub exact_ln: Option<f64>,
    pub exact_source: Option<String>,
    pub exact_skipped_too_large: bool,
    pub levels: Vec<LevelPoint>,
    pub phase1: Option<Phase1Summary>,
    pub plan: Option<PlanSummary>,
    pub sweep: Option<SweepProgress>,
    pub rounds: Vec<RoundPoint>,
    pub last_sample: Option<SampleSnapshot>,
    pub summary: Option<RunSummary>,
    pub error: Option<String>,
}

impl TelemetryState {
    pub fn reset_for_run(&mut self) {
        *self = Self { phase: Some(RunPhase::Init), ..Self::default() };
    }

    pub fn current_phase(&self) -> RunPhase {
        self.phase.unwrap_or(RunPhase::Idle)
    }

    pub fn is_running(&self) -> bool {
        matches!(self.current_phase(), RunPhase::Init | RunPhase::Learn | RunPhase::Plan | RunPhase::Estimate)
    }

    pub fn apply_event(&mut self, event: WebEvent) {
        match event {
            WebEvent::Init { n, nnz, engine, exact_ln, exact_source, exact_skipped_too_large, .. } => {
                self.phase = Some(RunPhase::Learn);
                self.n = n;
                self.nnz = nnz;
                self.engine = Some(engine);
                self.exact_ln = exact_ln;
                self.exact_source = exact_source;
                self.exact_skipped_too_large = exact_skipped_too_large;
            }
            WebEvent::Level { index, beta, step, relvar, log_w } => {
                self.phase = Some(RunPhase::Learn);
                self.levels.push(LevelPoint { index, beta, step, relvar, log_w });
            }
            WebEvent::Sample { beta, matching, valid } => {
                self.last_sample = Some(SampleSnapshot { beta, matching, valid });
            }
            WebEvent::Phase1(p1) => {
                self.phase = Some(RunPhase::Plan);
                self.phase1 = Some(p1);
            }
            WebEvent::Plan(plan) => {
                let seg_count = plan.segments.len();
                self.phase = Some(RunPhase::Estimate);
                self.plan = Some(plan);
                self.sweep = Some(SweepProgress { segment_done: vec![0; seg_count], ..SweepProgress::default() });
            }
            WebEvent::Sweep { round, done, total, segment, level, islands, .. } => {
                self.phase = Some(RunPhase::Estimate);
                let sweep = self.sweep.get_or_insert_with(SweepProgress::default);
                if sweep.round != round {
                    sweep.segment_done.fill(0);
                }
                sweep.round = round;
                sweep.done = done;
                sweep.total = total;
                sweep.segment = segment;
                sweep.level = level;
                sweep.islands = islands;
                if segment < sweep.segment_done.len() {
                    sweep.segment_done[segment] = sweep.segment_done[segment].max(level);
                }
            }
            WebEvent::Round { round, islands, ln_permanent, std_error, mutate_events, sampler_events, seconds } => {
                self.rounds.push(RoundPoint {
                    round,
                    islands,
                    ln_permanent,
                    std_error,
                    mutate_events,
                    sampler_events,
                    seconds,
                });
            }
            WebEvent::Done { summary } => {
                self.phase = Some(RunPhase::Done);
                if let Some(p1) = &summary.phase1 {
                    self.phase1 = Some(p1.clone());
                }
                if let Some(plan) = &summary.plan {
                    self.plan = Some(plan.clone());
                }
                self.exact_ln = summary.exact_ln;
                self.exact_source = summary.exact_source.clone();
                self.exact_skipped_too_large = summary.exact_skipped_too_large;
                self.summary = Some(*summary);
            }
            WebEvent::Cancelled => {
                self.phase = Some(RunPhase::Cancelled);
            }
            WebEvent::Failed { message } => {
                self.phase = Some(RunPhase::Failed);
                self.error = Some(message);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum WorkerStatus {
    Loading,
    Ready(Capabilities),
    Error(String),
}

pub fn format_compact_count(value: u64) -> String {
    if value >= 1_000_000_000 {
        format!("{:.2}B", value as f64 / 1e9)
    } else if value >= 1_000_000 {
        format!("{:.2}M", value as f64 / 1e6)
    } else if value >= 10_000 {
        format!("{:.1}k", value as f64 / 1e3)
    } else {
        value.to_string()
    }
}

pub fn format_sig(value: f64) -> String {
    if !value.is_finite() {
        return "—".into();
    }
    let abs = value.abs();
    if abs == 0.0 {
        "0".into()
    } else if !(0.01..1000.0).contains(&abs) {
        format!("{value:.2e}")
    } else if abs >= 100.0 {
        format!("{value:.0}")
    } else if abs >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}
