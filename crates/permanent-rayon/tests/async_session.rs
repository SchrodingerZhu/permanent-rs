use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, Waker};

use permanent_core::chain::stream::Stream as Philox;
use permanent_core::comm::Solo;
use permanent_core::engine::{AsyncEngine, Engine, FarRound, Stream};
use permanent_core::estimator::{AsyncSession, Cancelled, Config, Progress, Session};
use permanent_core::problem::Problem;
use permanent_core::sampler::{MutateCost, SampleCost, StageSchedule};
use permanent_core::scaling::{Prepared, Scaled, prepare};
use permanent_core::tables::Tables;
use permanent_rayon::Rayon;

/// Thin wrapper over `Rayon` reporting two concurrent lanes to exercise
/// threaded multi-lane sweeping without duplicating engine code.
struct TwoLane<'a>(Rayon<'a, f64, Philox>);

impl Engine for TwoLane<'_> {
    type Data = f64;
    type Counter = u64;

    fn name(&self) -> String {
        self.0.name()
    }

    fn capacity(&self) -> usize {
        self.0.capacity()
    }

    fn concurrency(&self, _particles: usize) -> usize {
        2
    }

    fn fork(&self) -> anyhow::Result<Self> {
        Ok(TwoLane(self.0.fork()?))
    }

    fn load(&mut self, perms: &[u32], first_id: u64) -> anyhow::Result<()> {
        self.0.load(perms, first_id)
    }

    fn anneal(
        &mut self,
        tables: &Tables<u64>,
        stages: &StageSchedule,
        stage_time: f64,
        max_trials: u64,
        count: usize,
        first_id: u64,
        stream: Stream,
    ) -> anyhow::Result<SampleCost> {
        self.0.anneal(tables, stages, stage_time, max_trials, count, first_id, stream)
    }

    fn mutate(&mut self, tables: &Tables<u64>, perfect_time: f64, stream: Stream) -> anyhow::Result<MutateCost> {
        self.0.mutate(tables, perfect_time, stream)
    }

    fn energies(&mut self) -> anyhow::Result<Vec<f64>> {
        self.0.energies()
    }

    fn log_gaps(&mut self) -> anyhow::Result<Vec<f64>> {
        self.0.log_gaps()
    }

    fn resample(&mut self, ancestors: &[u32]) -> anyhow::Result<()> {
        self.0.resample(ancestors)
    }

    fn far(
        &mut self,
        tables: &Tables<u64>,
        cycles: u64,
        max_steps: u64,
        groups: usize,
        stream: Stream,
    ) -> anyhow::Result<FarRound> {
        self.0.far(tables, cycles, max_steps, groups, stream)
    }

    fn head(&mut self, count: usize) -> anyhow::Result<Vec<u32>> {
        self.0.head(count)
    }
}

/// Future that yields `Pending` once before resolving, verifying `AsyncSession`
/// handles genuine suspension points.
async fn yield_once<T>(value: T) -> T {
    struct Yield(bool);
    impl Future for Yield {
        type Output = ();
        fn poll(mut self: std::pin::Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            if self.0 {
                Poll::Ready(())
            } else {
                self.0 = true;
                Poll::Pending
            }
        }
    }
    Yield(false).await;
    value
}

fn drive<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
            return output;
        }
    }
}

/// Adapts `Rayon` into a genuinely suspending `AsyncEngine`.
struct Yielding<'a>(Rayon<'a, f64, Philox>);

impl AsyncEngine for Yielding<'_> {
    type Data = f64;
    type Counter = u64;

    fn name(&self) -> String {
        self.0.name()
    }

    fn capacity(&self) -> usize {
        self.0.capacity()
    }

    fn load(&mut self, perms: &[u32], first_id: u64) -> anyhow::Result<()> {
        self.0.load(perms, first_id)
    }

    fn anneal(
        &mut self,
        tables: &Tables<u64>,
        stages: &StageSchedule,
        stage_time: f64,
        max_trials: u64,
        count: usize,
        first_id: u64,
        stream: Stream,
    ) -> impl Future<Output = anyhow::Result<SampleCost>> {
        let res = self.0.anneal(tables, stages, stage_time, max_trials, count, first_id, stream);
        yield_once(res)
    }

    fn mutate(
        &mut self,
        tables: &Tables<u64>,
        perfect_time: f64,
        stream: Stream,
    ) -> impl Future<Output = anyhow::Result<MutateCost>> {
        let res = self.0.mutate(tables, perfect_time, stream);
        yield_once(res)
    }

    fn energies(&mut self) -> impl Future<Output = anyhow::Result<Vec<f64>>> {
        let res = self.0.energies();
        yield_once(res)
    }

    fn log_gaps(&mut self) -> impl Future<Output = anyhow::Result<Vec<f64>>> {
        let res = self.0.log_gaps();
        yield_once(res)
    }

    fn resample(&mut self, ancestors: &[u32]) -> anyhow::Result<()> {
        self.0.resample(ancestors)
    }

    fn far(
        &mut self,
        tables: &Tables<u64>,
        cycles: u64,
        max_steps: u64,
        groups: usize,
        stream: Stream,
    ) -> impl Future<Output = anyhow::Result<FarRound>> {
        let res = self.0.far(tables, cycles, max_steps, groups, stream);
        yield_once(res)
    }

    fn head(&mut self, count: usize) -> impl Future<Output = anyhow::Result<Vec<u32>>> {
        let res = self.0.head(count);
        yield_once(res)
    }
}

fn tiny_scaled() -> Scaled {
    let problem = Problem {
        n: 4,
        a: vec![
            3.0, 1.0, 0.0, 0.5, //
            0.2, 2.0, 1.0, 0.0, //
            0.0, 0.7, 4.0, 1.0, //
            1.0, 0.0, 0.3, 2.5,
        ],
    };
    let Prepared::Scaled(scaled) = prepare(&problem) else { unreachable!() };
    scaled
}

fn tiny_config() -> Config {
    Config {
        epsilon: 0.5,
        delta: 0.05,
        seed: 7,
        learn_particles: Some(128),
        particles: Some(16),
        islands: Some(8),
        segments: Some(2),
        ..Default::default()
    }
}

/// `AsyncSession` (over genuine `Pending` yields) and synchronous `Session`
/// (both single-lane and multi-lane) must produce bit-for-bit identical
/// schedules and estimates for the same seed.
#[test]
fn async_and_sync_sessions_match_bit_for_bit() {
    let scaled = tiny_scaled();
    let config = tiny_config();

    let mut sync_one = Session::new(&scaled, &config, Rayon::<f64, Philox>::new(&scaled), &Solo).unwrap();
    let sync_levels = sync_one.learn_schedule().unwrap();
    let sync_plan = sync_one.plan(&sync_levels);
    let sync_est = sync_one.estimate(&sync_levels, &sync_plan).unwrap();

    let mut sync_two = Session::new(&scaled, &config, TwoLane(Rayon::<f64, Philox>::new(&scaled)), &Solo).unwrap();
    let sync2_levels = sync_two.learn_schedule().unwrap();
    let sync2_plan = sync_two.plan(&sync2_levels);
    let sync2_est = sync_two.estimate(&sync2_levels, &sync2_plan).unwrap();

    let mut samples = Vec::new();
    let mut async_session =
        AsyncSession::new(&scaled, &config, Yielding(Rayon::<f64, Philox>::new(&scaled)), &Solo).unwrap();
    async_session.capture_samples(true);
    async_session.observe(|progress| {
        if let Progress::Sample { beta, matching, valid } = progress {
            samples.push((beta, matching.to_vec(), valid));
        }
    });
    let (async_levels, async_est) = drive(async {
        let levels = async_session.learn_schedule().await.unwrap();
        let plan = async_session.plan(&levels);
        let est = async_session.estimate(&levels, &plan).await.unwrap();
        (levels, est)
    });
    drop(async_session);

    assert_eq!(sync_levels.len(), async_levels.len());
    for (a, b) in sync_levels.iter().zip(&async_levels) {
        assert_eq!(a.beta, b.beta);
        assert_eq!(a.log_w, b.log_w);
        assert_eq!(a.step, b.step);
    }
    assert_eq!(sync_est.ln_permanent, async_est.ln_permanent);
    assert_eq!(sync_est.std_error, async_est.std_error);
    assert_eq!(sync_est.ln_permanent, sync2_est.ln_permanent);
    assert_eq!(sync_est.std_error, sync2_est.std_error);

    // One sample per learned temperature plus at least one at beta = 1 from Phase 2.
    assert!(samples.len() > async_levels.len());
    for (_, matching, valid) in &samples {
        let mut sorted = matching.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..scaled.n as u32).collect::<Vec<_>>());
        assert_eq!(*valid, scaled.log_gap_of(matching).is_finite());
    }
    assert_eq!(samples.last().unwrap().0, 1.0);
}

#[test]
fn cancellation_stops_in_flight_run() {
    let scaled = tiny_scaled();
    let config = tiny_config();

    // Cancel during Phase 1 after the first learned temperature.
    let cancel_p1 = AtomicBool::new(false);
    let mut session = AsyncSession::new(&scaled, &config, Yielding(Rayon::<f64, Philox>::new(&scaled)), &Solo).unwrap();
    session.cancel_flag(&cancel_p1);
    session.observe(|progress| {
        if matches!(progress, Progress::Level { .. }) {
            cancel_p1.store(true, Ordering::Relaxed);
        }
    });
    let err = drive(session.learn_schedule()).unwrap_err();
    assert!(err.is::<Cancelled>(), "expected Cancelled, got {err:#}");

    // Cancel during Phase 2 after the first temperature sweep.
    let cancel_p2 = AtomicBool::new(false);
    let mut session = AsyncSession::new(&scaled, &config, Yielding(Rayon::<f64, Philox>::new(&scaled)), &Solo).unwrap();
    let levels = drive(session.learn_schedule()).unwrap();
    let plan = session.plan(&levels);
    session.cancel_flag(&cancel_p2);
    session.observe(|progress| {
        if matches!(progress, Progress::Sweep { .. }) {
            cancel_p2.store(true, Ordering::Relaxed);
        }
    });
    let err = drive(session.estimate(&levels, &plan)).unwrap_err();
    assert!(err.is::<Cancelled>(), "expected Cancelled, got {err:#}");
}
