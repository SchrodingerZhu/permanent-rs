//! Live dashboard for a run: progress with ETA, the temperature schedule as
//! Phase 1 builds it, the anchor weights learned at each temperature, the
//! per-temperature variance that sizes Phase 2, ln(permanent) as the
//! segments of Phase 2 build it, and the report as it is written.

// the dashboard's state is built on wasm, but nothing draws it there
#![cfg_attr(target_family = "wasm", allow(dead_code))]

use std::ops::Range;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use ratatui::Frame;
use ratatui::buffer::Buffer;
#[cfg(not(target_family = "wasm"))]
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Block, Chart, Dataset, Gauge, GraphType, Paragraph, Widget};

use permanent_core::estimator::Partial;
use permanent_core::stats::scientific;

/// Events streamed from the worker thread.
pub enum TuiEvent {
    Init {
        n: usize,
        engine: String,
        /// ln of the exact permanent, when it was computed.
        exact: Option<f64>,
    },
    /// Phase 1 fixed a temperature.
    Level {
        index: usize,
        beta: f64,
        relvar: f64,
        log_w: Vec<f64>,
    },
    /// Phase 2 is planned: the temperatures each segment multiplies, and the
    /// constant that makes the product of all the segments ln(permanent).
    Plan {
        segments: Vec<Range<usize>>,
        ln_scale: f64,
    },
    /// Phase 2 progress within one round of islands, in temperature sweeps,
    /// and where the sweep that just moved stands.
    Sweep {
        round: usize,
        done: u64,
        total: u64,
        partial: Partial,
    },
    /// One round of islands completed.
    Round {
        islands: usize,
        ln_permanent: f64,
        std_error: f64,
    },
    /// A line of the report.
    Line(String),
    Done,
    Failed(String),
}

/// What the dashboard hands back when it closes.
pub struct Outcome {
    /// The report lines received, to print once the terminal is restored.
    pub lines: Vec<String>,
    pub end: End,
}

pub enum End {
    Done,
    Failed(String),
    /// The user closed the dashboard while the run was still going.
    Quit,
}

#[derive(PartialEq)]
enum Phase {
    Loading,
    Learning,
    Sampling,
    Done,
    Failed(String),
}

struct SweepState {
    round: usize,
    done: u64,
    total: u64,
    started: Instant,
    /// When the round's last sweep finished; its rate is fixed from then on.
    ended: Option<Instant>,
}

impl SweepState {
    fn rate(&self) -> f64 {
        let seconds = (self.ended.unwrap_or_else(Instant::now) - self.started).as_secs_f64();
        self.done as f64 / seconds.max(1e-9)
    }
}

/// ln(e^a + e^b).
fn ln_add(a: f64, b: f64) -> f64 {
    let (high, low) = if a > b { (a, b) } else { (b, a) };
    if low == f64::NEG_INFINITY { high } else { high + (low - high).exp().ln_1p() }
}

/// The islands of a segment that got past one temperature, pooled over the
/// sweeps of a round: how many, and the logarithms of the sums of their
/// products so far and of the squares.
#[derive(Clone, Copy)]
struct Reached {
    islands: usize,
    ln_sum: f64,
    ln_sum_square: f64,
}

impl Reached {
    const NONE: Reached = Reached { islands: 0, ln_sum: f64::NEG_INFINITY, ln_sum_square: f64::NEG_INFINITY };

    fn add(&mut self, partial: &Partial) {
        let ln_islands = (partial.islands as f64).ln();
        self.islands += partial.islands;
        self.ln_sum = ln_add(self.ln_sum, ln_islands + partial.ln_mean);
        self.ln_sum_square = ln_add(self.ln_sum_square, ln_islands + partial.ln_mean_square);
    }

    /// ln of the mean product.
    fn ln_mean(&self) -> f64 {
        self.ln_sum - (self.islands as f64).ln()
    }

    /// Squared standard error of that logarithm, from the islands' spread.
    fn variance(&self) -> f64 {
        if self.islands < 2 {
            return 0.0;
        }
        let islands = self.islands as f64;
        let relative = (self.ln_sum_square + islands.ln() - 2.0 * self.ln_sum).exp() - 1.0;
        relative.max(0.0) / (islands - 1.0)
    }
}

/// What the sweeps of one segment have built in the current round. Entry k
/// of `reached` is temperature `steps.start + k`; the one at `steps.end`
/// closes the segment.
struct SegmentState {
    steps: Range<usize>,
    reached: Vec<Reached>,
}

impl SegmentState {
    /// The furthest temperature some sweep has passed, as an index into
    /// `reached`.
    fn head(&self) -> Option<usize> {
        self.reached.iter().rposition(|reached| reached.islands > 0)
    }

    fn through(&self) -> bool {
        self.head() == Some(self.steps.len())
    }
}

/// ln(permanent) as far as the current round has built it.
struct Built {
    ln_permanent: f64,
    std_error: f64,
    /// Segments whose sweeps have reached their end.
    through: usize,
}

struct App {
    phase: Phase,
    n: usize,
    engine: String,
    exact: Option<f64>,
    /// (level, log10 beta), skipping beta = 0.
    schedule: Vec<(f64, f64)>,
    /// (level, pilot relative variance of the level's ratio term).
    relvar: Vec<(f64, f64)>,
    /// Anchor weights ln w(r, v), one row per level.
    weights: Vec<Vec<f64>>,
    beta: f64,
    sweep: Option<SweepState>,
    /// The segments of Phase 2, in the order of their temperatures.
    segments: Vec<SegmentState>,
    ln_scale: f64,
    /// (islands, ln permanent, std error) after each completed round.
    rounds: Vec<(usize, f64, f64)>,
    lines: Vec<String>,
    started: Instant,
    finished: Option<Instant>,
}

impl App {
    fn new() -> Self {
        App {
            phase: Phase::Loading,
            n: 0,
            engine: String::new(),
            exact: None,
            schedule: Vec::new(),
            relvar: Vec::new(),
            weights: Vec::new(),
            beta: 0.0,
            sweep: None,
            segments: Vec::new(),
            ln_scale: 0.0,
            rounds: Vec::new(),
            lines: Vec::new(),
            started: Instant::now(),
            finished: None,
        }
    }

    fn update(&mut self, event: TuiEvent) {
        match event {
            TuiEvent::Init { n, engine, exact } => {
                self.n = n;
                self.engine = engine;
                self.exact = exact;
                self.phase = Phase::Learning;
            }
            TuiEvent::Level { index, beta, relvar, log_w } => {
                self.beta = beta;
                if beta > 0.0 {
                    self.schedule.push((index as f64, beta.log10()));
                }
                self.relvar.push((index as f64, relvar));
                self.weights.push(log_w);
            }
            TuiEvent::Plan { segments, ln_scale } => {
                self.segments = segments.into_iter().map(|steps| SegmentState { steps, reached: Vec::new() }).collect();
                self.ln_scale = ln_scale;
            }
            TuiEvent::Sweep { round, done, total, partial } => {
                self.phase = Phase::Sampling;
                let now = Instant::now();
                let (started, ended) = match &self.sweep {
                    Some(s) if s.round == round => (s.started, s.ended),
                    _ => {
                        // a round sweeps islands of its own: what it builds starts over
                        self.segments.iter_mut().for_each(|segment| segment.reached.clear());
                        (now, None)
                    }
                };
                let ended = ended.or((done >= total).then_some(now));
                self.sweep = Some(SweepState { round, done, total, started, ended });
                if let Some(segment) = self.segments.get_mut(partial.segment)
                    && let Some(passed) = partial.level.checked_sub(segment.steps.start)
                    && passed <= segment.steps.len()
                {
                    if segment.reached.len() <= passed {
                        segment.reached.resize(passed + 1, Reached::NONE);
                    }
                    segment.reached[passed].add(&partial);
                }
            }
            TuiEvent::Round { islands, ln_permanent, std_error } => {
                self.rounds.push((islands, ln_permanent, std_error))
            }
            TuiEvent::Line(line) => self.lines.push(line),
            TuiEvent::Done => {
                self.phase = Phase::Done;
                self.finished = Some(Instant::now());
            }
            TuiEvent::Failed(message) => {
                self.phase = Phase::Failed(message);
                self.finished = Some(Instant::now());
            }
        }
    }

    /// What the current round has built: every segment's product as far as
    /// its sweeps have got, times the constant. None before the first sweep.
    fn built(&self) -> Option<Built> {
        let mut built = Built { ln_permanent: self.ln_scale, std_error: 0.0, through: 0 };
        let mut variance = 0.0;
        let mut any = false;
        for segment in &self.segments {
            if let Some(head) = segment.head() {
                any = true;
                built.ln_permanent += segment.reached[head].ln_mean();
                variance += segment.reached[head].variance();
                built.through += segment.through() as usize;
            }
        }
        built.std_error = variance.sqrt();
        any.then_some(built)
    }

    /// One curve per segment: ln(permanent) as built up to each temperature
    /// its sweeps have passed, on top of what the segments before it stand
    /// at. The curves move while those are still being swept and join into
    /// one once every segment is through.
    fn curves(&self) -> Vec<Vec<(f64, f64)>> {
        let mut base = self.ln_scale;
        let mut curves = Vec::with_capacity(self.segments.len());
        for segment in &self.segments {
            let mut curve = vec![(segment.steps.start as f64, base)];
            let mut top = 0.0;
            for (passed, reached) in segment.reached.iter().enumerate() {
                if reached.islands > 0 {
                    // passing a temperature multiplies in the step to the next one
                    let x = (segment.steps.start + passed + 1).min(segment.steps.end);
                    top = reached.ln_mean();
                    curve.push((x as f64, base + top));
                }
            }
            base += top;
            curves.push(curve);
        }
        curves
    }

    /// Time since the start, frozen once the run has ended.
    fn elapsed(&self) -> Duration {
        self.finished.unwrap_or_else(Instant::now) - self.started
    }

    fn progress(&self) -> (f64, String) {
        match &self.phase {
            Phase::Loading => (0.0, "loading".into()),
            Phase::Learning => (
                self.beta,
                format!("phase 1: learning weights, {} temperatures, beta {:.4}", self.weights.len(), self.beta),
            ),
            Phase::Sampling => {
                let s = self.sweep.as_ref().unwrap();
                let ratio = s.done as f64 / s.total.max(1) as f64;
                let eta = if s.done > 0 && s.done < s.total {
                    let left = s.started.elapsed().mul_f64((s.total - s.done) as f64 / s.done as f64);
                    format!("  ETA {}", format_duration(left))
                } else {
                    String::new()
                };
                (ratio, format!("phase 2: round {}, {}/{} temperature sweeps{eta}", s.round + 1, s.done, s.total))
            }
            Phase::Done => (1.0, "done".into()),
            Phase::Failed(message) => (0.0, format!("failed: {message}")),
        }
    }
}

fn format_duration(d: Duration) -> String {
    let s = d.as_secs();
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

/// Half-block heatmap of the anchor weights: one matrix row per temperature
/// (top = beta 0), one column per right vertex. Each terminal cell shows two
/// rows through its foreground and background colors.
struct Heatmap<'a> {
    rows: &'a [Vec<f64>],
}

impl Heatmap<'_> {
    fn color(&self, row: usize, column: usize, min: f64, max: f64) -> Color {
        let value = self.rows[row][column];
        let v = if max > min { (value - min) / (max - min) } else { 0.5 };
        let c = (v.clamp(0.0, 1.0) * 255.0) as u8;
        Color::Rgb(c, 255 - c.abs_diff(128), 255 - c)
    }
}

impl Widget for Heatmap<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let levels = self.rows.len();
        if levels == 0 || area.width == 0 || area.height == 0 {
            return;
        }
        let n = self.rows[0].len();
        let (min, max) =
            self.rows.iter().flatten().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| (lo.min(x), hi.max(x)));
        let w = n.min(area.width as usize);
        let h = levels.div_ceil(2).min(area.height as usize);
        for y in 0..h {
            for x in 0..w {
                let column = x * n / w;
                let top = (2 * y) * levels / (2 * h);
                let bottom = ((2 * y + 1) * levels / (2 * h)).min(levels - 1);
                if let Some(cell) = buf.cell_mut((area.x + x as u16, area.y + y as u16)) {
                    cell.set_char('▀')
                        .set_fg(self.color(top, column, min, max))
                        .set_bg(self.color(bottom, column, min, max));
                }
            }
        }
    }
}

fn render(frame: &mut Frame, app: &App) {
    let [header, body, log, footer] =
        Layout::vertical([Constraint::Length(3), Constraint::Min(12), Constraint::Length(10), Constraint::Length(1)])
            .areas(frame.area());

    let (ratio, label) = app.progress();
    let gauge = Gauge::default()
        .block(Block::bordered().title(" permanent "))
        .gauge_style(Style::new().fg(Color::Green).bg(Color::DarkGray))
        .ratio(ratio.clamp(0.0, 1.0))
        .label(label);
    frame.render_widget(gauge, header);

    let [left, right] = Layout::horizontal([Constraint::Length(42), Constraint::Min(0)]).areas(body);
    let [stats_area, heatmap_area] = Layout::vertical([Constraint::Length(16), Constraint::Min(0)]).areas(left);
    // once Phase 2 is planned, what it builds takes the upper part and the
    // two charts of Phase 1 share a strip below
    let [schedule_area, relvar_area] = if app.segments.is_empty() {
        Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(right)
    } else {
        let rows = app.segments.len().min(SEGMENT_ROWS) as u16 + 2;
        let [built_area, segments_area, learned_area] =
            Layout::vertical([Constraint::Min(8), Constraint::Length(rows), Constraint::Length(8)]).areas(right);
        render_built(frame, app, built_area);
        render_segments(frame, app, segments_area);
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(learned_area)
    };

    render_stats(frame, app, stats_area);

    let heatmap_block = Block::bordered().title(" anchor weights ln w, by temperature ");
    let inner = heatmap_block.inner(heatmap_area);
    frame.render_widget(heatmap_block, heatmap_area);
    frame.render_widget(Heatmap { rows: &app.weights }, inner);

    render_chart(frame, schedule_area, " log10(beta) by temperature ", &app.schedule, Color::Cyan);
    render_chart(frame, relvar_area, " relative variance of each step's weight ", &app.relvar, Color::Magenta);

    let visible = log.height.saturating_sub(2) as usize;
    let tail = &app.lines[app.lines.len().saturating_sub(visible)..];
    frame.render_widget(
        Paragraph::new(tail.iter().map(|line| Line::from(line.as_str())).collect::<Vec<_>>())
            .block(Block::bordered().title(" report ")),
        log,
    );

    frame.render_widget(Line::from(vec![Span::raw(" q ").bold(), Span::raw("quit")]), footer);
}

fn render_stats(frame: &mut Frame, app: &App, area: Rect) {
    let phase = match app.phase {
        Phase::Loading => "loading",
        Phase::Learning => "learning weights",
        Phase::Sampling => "estimating",
        Phase::Done => "done",
        Phase::Failed(_) => "failed",
    };
    let spinner = if app.phase == Phase::Sampling { spinner(app) } else { "" };
    let mut lines = vec![
        Line::from(vec![Span::raw("phase         "), Span::raw(phase).bold(), Span::raw(" "), Span::raw(spinner)]),
        Line::from(format!("n             {}", app.n)),
        Line::from(format!("engine        {}", app.engine)),
        Line::from(format!("temperatures  {}", app.weights.len())),
    ];
    if let Some(s) = &app.sweep {
        lines.push(Line::from(format!("sweeps/s      {:.1}", s.rate())));
    }
    if let Some(built) = app.built() {
        // not an estimate until every segment is through: say how far it is
        let all = built.through == app.segments.len();
        let label = if all { "this round    " } else { "built so far  " };
        let style = if all { Style::new() } else { Style::new().fg(Color::DarkGray) };
        lines.push(Line::from(format!("segments      {} of {} through", built.through, app.segments.len())));
        lines.push(Line::from(vec![
            Span::raw(label),
            Span::styled(format!("{:.5} ± {:.5}", built.ln_permanent, built.std_error), style),
        ]));
    }
    if let Some(&(islands, ln_permanent, std_error)) = app.rounds.last() {
        lines.push(Line::from(format!("islands       {islands}")));
        lines.push(Line::from(format!("ln(per)       {ln_permanent:.5} ± {std_error:.5}")));
        let style = if app.phase == Phase::Done { Style::new().fg(Color::Green).bold() } else { Style::new() };
        lines.push(Line::from(vec![Span::raw("permanent     "), Span::styled(scientific(ln_permanent), style)]));
        if let Some(exact) = app.exact {
            let error = 100.0 * ((ln_permanent - exact).exp() - 1.0);
            let style = if error.abs() < 5.0 { Style::new().fg(Color::Green) } else { Style::new().fg(Color::Yellow) };
            lines.push(Line::from(format!("exact         {}", scientific(exact))));
            lines.push(Line::from(vec![Span::raw("rel error     "), Span::styled(format!("{error:+.2}%"), style)]));
        }
    }
    lines.push(Line::from(format!("elapsed       {}", format_duration(app.elapsed()))));
    frame.render_widget(Paragraph::new(lines).block(Block::bordered().title(" stats ")), area);
}

/// Segments listed one to a row; a run rarely has more.
const SEGMENT_ROWS: usize = 8;
/// Colors of the segments, in turn, in the chart and in their rows.
const SEGMENT_COLORS: [Color; 6] =
    [Color::Cyan, Color::Yellow, Color::Magenta, Color::Green, Color::LightBlue, Color::LightRed];
/// Cells of a segment's bar, and the eighths a cell fills in.
const BAR_CELLS: usize = 24;
const EIGHTHS: [&str; 8] = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉"];
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// A mark that turns while the run is going.
fn spinner(app: &App) -> &'static str {
    SPINNER[(app.elapsed().as_millis() / 100) as usize % SPINNER.len()]
}

/// ln(permanent) as the segments build it, one curve each.
fn render_built(frame: &mut Frame, app: &App, area: Rect) {
    let title = " ln(permanent) built up along the temperatures: partial until every segment is through ";
    let curves = app.curves();
    let steps = app.segments.last().map_or(1, |segment| segment.steps.end).max(1) as f64;
    let (y_min, y_max) =
        curves.iter().flatten().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| (lo.min(p.1), hi.max(p.1)));
    if !y_min.is_finite() {
        frame.render_widget(Block::bordered().title(title), area);
        return;
    }
    let margin = ((y_max - y_min) * 0.05).max(0.01);
    let (y_min, y_max) = (y_min - margin, y_max + margin);
    let datasets = curves
        .iter()
        .enumerate()
        .map(|(segment, curve)| {
            Dataset::default()
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::new().fg(SEGMENT_COLORS[segment % SEGMENT_COLORS.len()]))
                .data(curve)
        })
        .collect::<Vec<_>>();
    let chart = Chart::new(datasets)
        .block(Block::bordered().title(title))
        .x_axis(Axis::default().bounds([0.0, steps]).labels(["0".to_string(), format!("{steps:.0}")]))
        .y_axis(Axis::default().bounds([y_min, y_max]).labels([format!("{y_min:.2}"), format!("{y_max:.2}")]));
    frame.render_widget(chart, area);
}

/// One row per segment: how far its sweeps are and what they have built.
fn render_segments(frame: &mut Frame, app: &App, area: Rect) {
    let lines: Vec<Line> = app
        .segments
        .iter()
        .enumerate()
        .take(SEGMENT_ROWS)
        .map(|(index, segment)| {
            let color = Style::new().fg(SEGMENT_COLORS[index % SEGMENT_COLORS.len()]);
            let sweeps = segment.steps.len() + 1;
            let passed = segment.head().map_or(0, |head| head + 1);
            let eighths = passed * BAR_CELLS * 8 / sweeps;
            let mut bar = "█".repeat(eighths / 8);
            if eighths / 8 < BAR_CELLS {
                bar.push_str(EIGHTHS[eighths % 8]);
                bar.push_str(&" ".repeat(BAR_CELLS - eighths / 8 - 1));
            }
            let built = match segment.head() {
                Some(head) => {
                    let reached = &segment.reached[head];
                    let mark = if segment.through() { "✓" } else { spinner(app) };
                    format!(
                        " {mark} {:+.5} ± {:.5}  {} islands",
                        reached.ln_mean(),
                        reached.variance().sqrt(),
                        reached.islands
                    )
                }
                None => "   waiting".to_string(),
            };
            Line::from(vec![
                Span::styled(
                    format!(" {:>2}  {:>4}..{:<4} ", index + 1, segment.steps.start, segment.steps.end),
                    color,
                ),
                Span::styled(bar, color.bg(Color::DarkGray)),
                Span::raw(format!(" {passed:>4}/{sweeps:<4}")),
                Span::raw(built),
            ])
        })
        .collect();
    let title = " segments: temperatures, sweeps done, ln of the ratio built so far ";
    frame.render_widget(Paragraph::new(lines).block(Block::bordered().title(title)), area);
}

fn render_chart(frame: &mut Frame, area: Rect, title: &str, data: &[(f64, f64)], color: Color) {
    if data.is_empty() {
        frame.render_widget(Block::bordered().title(title), area);
        return;
    }
    let x_max = data.last().map(|p| p.0).unwrap_or(1.0).max(1.0);
    let (y_min, y_max) = data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| (lo.min(p.1), hi.max(p.1)));
    let margin = ((y_max - y_min) * 0.05).max(0.01);
    let (y_min, y_max) = (y_min - margin, y_max + margin);
    let dataset = Dataset::default()
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::new().fg(color))
        .data(data);
    let chart = Chart::new(vec![dataset])
        .block(Block::bordered().title(title))
        .x_axis(Axis::default().bounds([0.0, x_max]).labels(["0".to_string(), format!("{x_max:.0}")]))
        .y_axis(Axis::default().bounds([y_min, y_max]).labels([format!("{y_min:.2}"), format!("{y_max:.2}")]));
    frame.render_widget(chart, area);
}

/// There is no terminal to take over on wasm.
#[cfg(target_family = "wasm")]
pub fn run(_: Receiver<TuiEvent>) -> std::io::Result<Outcome> {
    Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "the dashboard is not available on wasm"))
}

/// Apply the events received so far. A worker that is gone without having
/// sent `Done` or `Failed` panicked, which fails the run.
fn drain(app: &mut App, rx: &Receiver<TuiEvent>) {
    loop {
        match rx.try_recv() {
            Ok(event) => app.update(event),
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => {
                if !matches!(app.phase, Phase::Done | Phase::Failed(_)) {
                    app.update(TuiEvent::Failed("the run ended without a result".into()));
                }
                return;
            }
        }
    }
}

/// Run the dashboard until the user quits. The run keeps going behind it;
/// the caller decides what quitting early means.
#[cfg(not(target_family = "wasm"))]
pub fn run(rx: Receiver<TuiEvent>) -> std::io::Result<Outcome> {
    let mut terminal = ratatui::init();
    let mut app = App::new();
    let result = loop {
        drain(&mut app, &rx);
        if let Err(error) = terminal.draw(|frame| render(frame, &app)) {
            break Err(error);
        }
        match event::poll(Duration::from_millis(50)) {
            Ok(true) => {}
            Ok(false) => continue,
            Err(error) => break Err(error),
        }
        match event::read() {
            Ok(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                let quit = matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
                    || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL));
                if quit {
                    break Ok(());
                }
            }
            Ok(_) => {}
            Err(error) => break Err(error),
        }
    };
    ratatui::restore();
    result?;
    let end = match app.phase {
        Phase::Done => End::Done,
        Phase::Failed(message) => End::Failed(message),
        _ => End::Quit,
    };
    Ok(Outcome { lines: app.lines, end })
}

#[cfg(test)]
mod test {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn screen(app: &App) -> String {
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        terminal.draw(|frame| render(frame, app)).unwrap();
        let buffer = terminal.backend().buffer();
        buffer.content().iter().map(|cell| cell.symbol()).collect()
    }

    #[test]
    fn every_phase_renders() {
        let mut app = App::new();
        assert!(screen(&app).contains("loading"));
        app.update(TuiEvent::Init { n: 4, engine: "cpu".into(), exact: Some(3.0) });
        for index in 0..5 {
            app.update(TuiEvent::Level {
                index,
                beta: index as f64 / 4.0,
                relvar: 0.2,
                log_w: vec![1.0, 1.5, 0.5, 1.2],
            });
        }
        assert!(screen(&app).contains("phase 1: learning weights, 5 temperatures"));
        app.update(TuiEvent::Plan { segments: vec![0..2, 2..4], ln_scale: 1.0 });
        app.update(TuiEvent::Sweep { round: 0, done: 10, total: 40, partial: partial(1, 2, 8, 0.5) });
        let text = screen(&app);
        assert!(text.contains("10/40 temperature sweeps"), "{text}");
        assert!(text.contains("segments      0 of 2 through"), "{text}");
        assert!(text.contains("built so far  1.50000"), "{text}");
        // a full bar stops the rate, whatever the run does next
        app.update(TuiEvent::Sweep { round: 0, done: 40, total: 40, partial: partial(1, 4, 8, 0.75) });
        let rate = app.sweep.as_ref().unwrap().rate();
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(app.sweep.as_ref().unwrap().rate(), rate);
        // and the next round starts its own clock
        app.update(TuiEvent::Sweep { round: 1, done: 1, total: 40, partial: partial(0, 0, 8, 0.25) });
        assert!(app.sweep.as_ref().unwrap().ended.is_none());
        // and builds on its own islands only
        assert!(screen(&app).contains("built so far  1.25000"));
        app.update(TuiEvent::Round { islands: 32, ln_permanent: 3.01, std_error: 0.04 });
        app.update(TuiEvent::Line("permanent ≈ 2.028740e+1".into()));
        app.update(TuiEvent::Done);
        let text = screen(&app);
        assert!(text.contains("3.01000 ± 0.04000"), "{text}");
        assert!(text.contains("rel error     +1.01%"), "{text}");
        assert!(text.contains("permanent ≈ 2.028740e+1"), "{text}");
    }

    /// A sweep of `islands` islands that all stand at `ln_ratio`.
    fn partial(segment: usize, level: usize, islands: usize, ln_ratio: f64) -> Partial {
        Partial { segment, level, islands, ln_mean: ln_ratio, ln_mean_square: 2.0 * ln_ratio }
    }

    /// The sweeps of a segment pool as if their islands were swept together,
    /// and the segments add up to ln(permanent) once all are through.
    #[test]
    fn sweeps_pool_and_segments_add_up() {
        let mut app = App::new();
        app.update(TuiEvent::Plan { segments: vec![0..1, 1..2], ln_scale: 10.0 });
        // segment 0: islands at ratios 1, 3 and 5, swept as one of two and one of one
        let two = Partial { segment: 0, level: 1, islands: 2, ln_mean: 2f64.ln(), ln_mean_square: 5f64.ln() };
        let one = Partial { segment: 0, level: 1, islands: 1, ln_mean: 5f64.ln(), ln_mean_square: 25f64.ln() };
        app.update(TuiEvent::Sweep { round: 0, done: 1, total: 4, partial: two });
        app.update(TuiEvent::Sweep { round: 0, done: 2, total: 4, partial: one });
        let built = app.built().unwrap();
        assert_eq!(built.through, 1);
        assert!((built.ln_permanent - (10.0 + 3f64.ln())).abs() < 1e-12);
        // relative variance of 1, 3, 5 around their mean 3 is 8/27; over 3 - 1 islands
        assert!((built.std_error - (8.0f64 / 27.0 / 2.0).sqrt()).abs() < 1e-12);
        app.update(TuiEvent::Sweep { round: 0, done: 3, total: 4, partial: partial(1, 1, 4, 0.5) });
        app.update(TuiEvent::Sweep { round: 0, done: 4, total: 4, partial: partial(1, 2, 4, 2.0) });
        let built = app.built().unwrap();
        assert_eq!(built.through, 2);
        assert!((built.ln_permanent - (10.0 + 3f64.ln() + 2.0)).abs() < 1e-12);
        // the second curve starts where the first one stands and ends at the total
        let curves = app.curves();
        assert_eq!(curves[1][0].0, 1.0);
        assert!((curves[1][0].1 - (10.0 + 3f64.ln())).abs() < 1e-12);
        assert!((curves[1].last().unwrap().1 - built.ln_permanent).abs() < 1e-12);
        assert!(screen(&app).contains("this round    13.09861"));
    }

    #[test]
    fn a_worker_that_vanishes_fails_the_run() {
        let (sender, receiver) = std::sync::mpsc::channel();
        sender.send(TuiEvent::Init { n: 4, engine: "cpu".into(), exact: None }).unwrap();
        drop(sender);
        let mut app = App::new();
        drain(&mut app, &receiver);
        assert!(matches!(app.phase, Phase::Failed(_)));

        // one that finished first is not affected
        let (sender, receiver) = std::sync::mpsc::channel();
        sender.send(TuiEvent::Done).unwrap();
        drop(sender);
        let mut app = App::new();
        drain(&mut app, &receiver);
        assert!(matches!(app.phase, Phase::Done));
    }
}
