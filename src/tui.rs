//! Live dashboard for a run: progress with ETA, the temperature schedule as
//! Phase 1 builds it, the anchor weights learned at each temperature, the
//! per-temperature variance that sizes Phase 2, and the report as it is
//! written.

use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Block, Chart, Dataset, Gauge, GraphType, Paragraph, Widget};

use crate::stats::scientific;

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
    /// Phase 2 progress within one round of islands, in temperature sweeps.
    Sweep {
        round: usize,
        done: u64,
        total: u64,
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
            TuiEvent::Sweep { round, done, total } => {
                self.phase = Phase::Sampling;
                let started = match &self.sweep {
                    Some(s) if s.round == round => s.started,
                    _ => Instant::now(),
                };
                self.sweep = Some(SweepState { round, done, total, started });
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
        let (min, max) = self
            .rows
            .iter()
            .flatten()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| (lo.min(x), hi.max(x)));
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
    let [header, body, log, footer] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(12),
        Constraint::Length(10),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    let (ratio, label) = app.progress();
    let gauge = Gauge::default()
        .block(Block::bordered().title(" permanent "))
        .gauge_style(Style::new().fg(Color::Green).bg(Color::DarkGray))
        .ratio(ratio.clamp(0.0, 1.0))
        .label(label);
    frame.render_widget(gauge, header);

    let [left, right] = Layout::horizontal([Constraint::Length(42), Constraint::Min(0)]).areas(body);
    let [stats_area, heatmap_area] = Layout::vertical([Constraint::Length(14), Constraint::Min(0)]).areas(left);
    let [schedule_area, relvar_area] =
        Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(right);

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

    frame.render_widget(
        Line::from(vec![Span::raw(" q ").bold(), Span::raw("quit")]),
        footer,
    );
}

fn render_stats(frame: &mut Frame, app: &App, area: Rect) {
    let phase = match app.phase {
        Phase::Loading => "loading",
        Phase::Learning => "learning weights",
        Phase::Sampling => "estimating",
        Phase::Done => "done",
        Phase::Failed(_) => "failed",
    };
    let mut lines = vec![
        Line::from(vec![Span::raw("phase         "), Span::raw(phase).bold()]),
        Line::from(format!("n             {}", app.n)),
        Line::from(format!("engine        {}", app.engine)),
        Line::from(format!("temperatures  {}", app.weights.len())),
    ];
    if let Some(s) = &app.sweep {
        let rate = s.done as f64 / s.started.elapsed().as_secs_f64().max(1e-9);
        if app.phase == Phase::Sampling {
            lines.push(Line::from(format!("sweeps/s      {rate:.1}")));
        }
    }
    if let Some(&(islands, ln_permanent, std_error)) = app.rounds.last() {
        lines.push(Line::from(format!("islands       {islands}")));
        lines.push(Line::from(format!("ln(per)       {ln_permanent:.5} ± {std_error:.5}")));
        let style = if app.phase == Phase::Done { Style::new().fg(Color::Green).bold() } else { Style::new() };
        lines.push(Line::from(vec![
            Span::raw("permanent     "),
            Span::styled(scientific(ln_permanent), style),
        ]));
        if let Some(exact) = app.exact {
            let error = 100.0 * ((ln_permanent - exact).exp() - 1.0);
            let style = if error.abs() < 5.0 { Style::new().fg(Color::Green) } else { Style::new().fg(Color::Yellow) };
            lines.push(Line::from(format!("exact         {}", scientific(exact))));
            lines.push(Line::from(vec![
                Span::raw("rel error     "),
                Span::styled(format!("{error:+.2}%"), style),
            ]));
        }
    }
    lines.push(Line::from(format!("elapsed       {}", format_duration(app.elapsed()))));
    frame.render_widget(Paragraph::new(lines).block(Block::bordered().title(" stats ")), area);
}

fn render_chart(frame: &mut Frame, area: Rect, title: &str, data: &[(f64, f64)], color: Color) {
    if data.is_empty() {
        frame.render_widget(Block::bordered().title(title), area);
        return;
    }
    let x_max = data.last().map(|p| p.0).unwrap_or(1.0).max(1.0);
    let (y_min, y_max) =
        data.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| (lo.min(p.1), hi.max(p.1)));
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

/// Run the dashboard until the user quits. The run keeps going behind it;
/// the caller decides what quitting early means.
pub fn run(rx: Receiver<TuiEvent>) -> std::io::Result<Outcome> {
    let mut terminal = ratatui::init();
    let mut app = App::new();
    let result = loop {
        while let Ok(event) = rx.try_recv() {
            app.update(event);
        }
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
        app.update(TuiEvent::Sweep { round: 0, done: 10, total: 40 });
        assert!(screen(&app).contains("10/40 temperature sweeps"));
        app.update(TuiEvent::Round { islands: 32, ln_permanent: 3.01, std_error: 0.04 });
        app.update(TuiEvent::Line("permanent ≈ 2.028740e+1".into()));
        app.update(TuiEvent::Done);
        let text = screen(&app);
        assert!(text.contains("3.01000 ± 0.04000"), "{text}");
        assert!(text.contains("rel error     +1.01%"), "{text}");
        assert!(text.contains("permanent ≈ 2.028740e+1"), "{text}");
    }
}
