//! Live dashboard for the annealing run, in the spirit of burn.dev's
//! training visualization: overall progress with ETA, live charts of the
//! estimator and the per-step ratios, and a heatmap of the edge marginals
//! (the "steady state" matrix) converging onto the permanent's support.

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

/// Events streamed from the annealing worker thread.
pub enum TuiEvent {
    /// Graph loaded; adjacency is row-major n x n.
    Init {
        n: usize,
        adjacency: Vec<bool>,
    },
    WarmupStarted,
    Step(StepUpdate),
    Done {
        estimator: f64,
    },
}

pub struct StepUpdate {
    pub step: usize,
    pub total_steps: usize,
    pub beta: f64,
    pub ratio: f64,
    pub estimator: f64,
    /// fraction of rejection-sampling attempts that produced a sample
    pub acceptance: f64,
    /// row-major n x n edge marginals (1 / w)
    pub marginals: Vec<f64>,
}

enum Phase {
    Loading,
    Warmup,
    Running,
    Done,
}

struct App {
    phase: Phase,
    n: usize,
    adjacency: Vec<bool>,
    latest: Option<StepUpdate>,
    /// (step, log10 estimator) per step
    estimator_history: Vec<(f64, f64)>,
    /// (step, ratio) per step
    ratio_history: Vec<(f64, f64)>,
    exact: Option<f64>,
    final_estimator: Option<f64>,
    started: Instant,
    cooling_started: Option<Instant>,
}

impl App {
    fn new(exact: Option<f64>) -> Self {
        App {
            phase: Phase::Loading,
            n: 0,
            adjacency: Vec::new(),
            latest: None,
            estimator_history: Vec::new(),
            ratio_history: Vec::new(),
            exact,
            final_estimator: None,
            started: Instant::now(),
            cooling_started: None,
        }
    }

    fn update(&mut self, event: TuiEvent) {
        match event {
            TuiEvent::Init { n, adjacency } => {
                self.n = n;
                self.adjacency = adjacency;
            }
            TuiEvent::WarmupStarted => self.phase = Phase::Warmup,
            TuiEvent::Step(update) => {
                if self.cooling_started.is_none() {
                    self.cooling_started = Some(Instant::now());
                }
                self.phase = Phase::Running;
                self.estimator_history
                    .push((update.step as f64, update.estimator.log10()));
                self.ratio_history.push((update.step as f64, update.ratio));
                self.latest = Some(update);
            }
            TuiEvent::Done { estimator } => {
                self.phase = Phase::Done;
                self.final_estimator = Some(estimator);
            }
        }
    }

    fn eta(&self) -> Option<Duration> {
        let latest = self.latest.as_ref()?;
        let elapsed = self.cooling_started?.elapsed();
        if latest.step == 0 {
            return None;
        }
        let per_step = elapsed.div_f64(latest.step as f64);
        Some(per_step.mul_f64((latest.total_steps - latest.step) as f64))
    }
}

fn format_duration(d: Duration) -> String {
    let s = d.as_secs();
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

/// Half-block heatmap of the edge marginals: each terminal cell renders two
/// matrix rows via foreground (upper) and background (lower) colors. Graph
/// edges shade black -> green, non-edges black -> red, so a converged run
/// shows the permanent's support in green with the red mass faded out.
struct Heatmap<'a> {
    marginals: &'a [f64],
    adjacency: &'a [bool],
    n: usize,
}

impl Heatmap<'_> {
    fn color(&self, i: usize, j: usize, max: f64) -> Color {
        let q = self.marginals[i * self.n + j];
        let v = if max > 0.0 {
            (q / max).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let v = v.sqrt(); // gamma boost for small values
        let c = (v * 255.0) as u8;
        if self.adjacency[i * self.n + j] {
            Color::Rgb(c / 4, c, c / 2)
        } else {
            Color::Rgb(c, c / 5, c / 5)
        }
    }
}

impl Widget for Heatmap<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if self.n == 0 || area.width == 0 || area.height == 0 {
            return;
        }
        let max = self.marginals.iter().copied().fold(0.0f64, f64::max);
        let w = (self.n as u16).min(area.width) as usize;
        let h = (self.n as u16).div_ceil(2).min(area.height) as usize;
        for y in 0..h {
            for x in 0..w {
                let j = x * self.n / w;
                let i_top = (2 * y) * self.n / (2 * h);
                let i_bottom = ((2 * y + 1) * self.n / (2 * h)).min(self.n - 1);
                let top = self.color(i_top, j, max);
                let bottom = self.color(i_bottom, j, max);
                if let Some(cell) = buf.cell_mut((area.x + x as u16, area.y + y as u16)) {
                    cell.set_char('▀').set_fg(top).set_bg(bottom);
                }
            }
        }
    }
}

fn render(frame: &mut Frame, app: &App) {
    let [header, body, footer] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    // header: overall progress
    let (progress, label) = match (&app.phase, &app.latest) {
        (Phase::Loading, _) => (0.0, "loading graph...".to_string()),
        (Phase::Warmup, _) => (0.0, "warming up chains...".to_string()),
        (_, Some(l)) => {
            let ratio = l.step as f64 / l.total_steps as f64;
            let eta = match (&app.phase, app.eta()) {
                (Phase::Done, _) => "done".to_string(),
                (_, Some(eta)) => format!("ETA {}", format_duration(eta)),
                _ => String::new(),
            };
            (
                ratio,
                format!("{}/{} cooling steps  {}", l.step, l.total_steps, eta),
            )
        }
        _ => (0.0, String::new()),
    };
    let gauge = Gauge::default()
        .block(Block::bordered().title(" permanent-rs — simulated annealing "))
        .gauge_style(Style::new().fg(Color::Green).bg(Color::DarkGray))
        .ratio(progress.clamp(0.0, 1.0))
        .label(label);
    frame.render_widget(gauge, header);

    let [left, right] =
        Layout::horizontal([Constraint::Length(38), Constraint::Min(0)]).areas(body);
    let [stats_area, heatmap_area] =
        Layout::vertical([Constraint::Length(13), Constraint::Min(0)]).areas(left);
    let [estimator_area, ratio_area] =
        Layout::vertical([Constraint::Percentage(60), Constraint::Percentage(40)]).areas(right);

    render_stats(frame, app, stats_area);

    let heatmap_block = Block::bordered().title(" edge marginals 1/w ");
    let inner = heatmap_block.inner(heatmap_area);
    frame.render_widget(heatmap_block, heatmap_area);
    if let Some(latest) = &app.latest {
        frame.render_widget(
            Heatmap {
                marginals: &latest.marginals,
                adjacency: &app.adjacency,
                n: app.n,
            },
            inner,
        );
    }

    render_estimator_chart(frame, app, estimator_area);
    render_ratio_chart(frame, app, ratio_area);

    frame.render_widget(
        Line::from(vec![
            Span::raw(" q ").bold(),
            Span::raw("quit  "),
            Span::raw("(annealing keeps running until quit or done)").dark_gray(),
        ]),
        footer,
    );
}

fn render_stats(frame: &mut Frame, app: &App, area: Rect) {
    let mut lines: Vec<Line> = Vec::new();
    let phase = match app.phase {
        Phase::Loading => "loading",
        Phase::Warmup => "warmup",
        Phase::Running => "cooling",
        Phase::Done => "done",
    };
    lines.push(Line::from(vec![
        Span::raw("phase      "),
        Span::raw(phase).bold(),
    ]));
    lines.push(Line::from(format!("n          {}", app.n)));
    if let Some(l) = &app.latest {
        lines.push(Line::from(format!("beta       {:.4}", l.beta)));
        lines.push(Line::from(format!("ratio      {:.6}", l.ratio)));
        lines.push(Line::from(format!("estimator  {:.6e}", l.estimator)));
        lines.push(Line::from(format!("log10 est  {:.4}", l.estimator.log10())));
        if let Some(exact) = app.exact {
            let err = (l.estimator - exact).abs() / exact * 100.0;
            let style = if err < 5.0 {
                Style::new().fg(Color::Green)
            } else {
                Style::new().fg(Color::Yellow)
            };
            lines.push(Line::from(format!("exact      {exact:.6e}")));
            lines.push(Line::from(vec![
                Span::raw("rel error  "),
                Span::styled(format!("{err:.2}%"), style),
            ]));
        }
        lines.push(Line::from(format!(
            "accept     {:.1}%",
            l.acceptance * 100.0
        )));
        if let Some(start) = app.cooling_started {
            let steps_per_sec = l.step as f64 / start.elapsed().as_secs_f64().max(1e-9);
            lines.push(Line::from(format!("steps/s    {steps_per_sec:.2}")));
        }
    }
    lines.push(Line::from(format!(
        "elapsed    {}",
        format_duration(app.started.elapsed())
    )));
    if let Some(estimator) = app.final_estimator {
        lines.push(Line::from(vec![
            Span::raw("final      "),
            Span::styled(
                format!("{estimator:.6e}"),
                Style::new().fg(Color::Green).bold(),
            ),
        ]));
    }
    frame.render_widget(
        Paragraph::new(lines).block(Block::bordered().title(" stats ")),
        area,
    );
}

fn render_estimator_chart(frame: &mut Frame, app: &App, area: Rect) {
    if app.estimator_history.is_empty() {
        frame.render_widget(Block::bordered().title(" log10(estimator) "), area);
        return;
    }
    let total = app
        .latest
        .as_ref()
        .map(|l| l.total_steps as f64)
        .unwrap_or(1.0);
    let ys = app.estimator_history.iter().map(|p| p.1);
    let mut y_min = ys.clone().fold(f64::INFINITY, f64::min);
    let mut y_max = ys.fold(f64::NEG_INFINITY, f64::max);
    let target = app.exact.map(|e| e.log10());
    let target_line = target.map(|t| [(0.0, t), (total, t)]);
    if let Some(t) = target {
        y_min = y_min.min(t);
        y_max = y_max.max(t);
    }
    let margin = ((y_max - y_min) * 0.05).max(0.1);
    let (y_min, y_max) = (y_min - margin, y_max + margin);
    let mut datasets = vec![
        Dataset::default()
            .name("log10 estimator")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::new().fg(Color::Cyan))
            .data(&app.estimator_history),
    ];
    if let Some(line) = &target_line {
        datasets.push(
            Dataset::default()
                .name("log10 exact")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::new().fg(Color::Green))
                .data(line),
        );
    }
    let chart = Chart::new(datasets)
        .block(Block::bordered().title(" log10(estimator) "))
        .x_axis(
            Axis::default()
                .bounds([0.0, total])
                .labels(["0".to_string(), format!("{total:.0}")]),
        )
        .y_axis(
            Axis::default()
                .bounds([y_min, y_max])
                .labels([format!("{y_min:.1}"), format!("{y_max:.1}")]),
        );
    frame.render_widget(chart, area);
}

fn render_ratio_chart(frame: &mut Frame, app: &App, area: Rect) {
    const WINDOW: usize = 512;
    let history = &app.ratio_history;
    if history.is_empty() {
        frame.render_widget(Block::bordered().title(" step ratio "), area);
        return;
    }
    let window = &history[history.len().saturating_sub(WINDOW)..];
    let x_min = window.first().map(|p| p.0).unwrap_or(0.0);
    let x_max = window.last().map(|p| p.0).unwrap_or(1.0).max(x_min + 1.0);
    let y_min = window.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let y_min = (y_min - 0.01).min(0.99);
    let dataset = Dataset::default()
        .name(format!("ratio (last {})", window.len()))
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::new().fg(Color::Magenta))
        .data(window);
    let chart = Chart::new(vec![dataset])
        .block(Block::bordered().title(" step ratio Z(b')/Z(b) "))
        .x_axis(
            Axis::default()
                .bounds([x_min, x_max])
                .labels([format!("{x_min:.0}"), format!("{x_max:.0}")]),
        )
        .y_axis(
            Axis::default()
                .bounds([y_min, 1.01])
                .labels([format!("{y_min:.3}"), "1.0".to_string()]),
        );
    frame.render_widget(chart, area);
}

/// Run the dashboard until the user quits. Returns the final estimator if
/// the annealing completed while the dashboard was open.
pub fn run(rx: Receiver<TuiEvent>, exact: Option<f64>) -> std::io::Result<Option<f64>> {
    let mut terminal = ratatui::init();
    let mut app = App::new(exact);
    let result = loop {
        while let Ok(event) = rx.try_recv() {
            app.update(event);
        }
        terminal.draw(|frame| render(frame, &app))?;
        if event::poll(Duration::from_millis(50))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            let quit = matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
                || (key.code == KeyCode::Char('c')
                    && key.modifiers.contains(KeyModifiers::CONTROL));
            if quit {
                break app.final_estimator;
            }
        }
    };
    ratatui::restore();
    Ok(result)
}
