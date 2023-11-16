use std::num::NonZeroUsize;

struct CoolingConfig {
    n: NonZeroUsize,
    additive_ratio: NonZeroUsize,
    multiplicative_ratio: NonZeroUsize,
}

impl CoolingConfig {
    fn log(&self) -> usize {
        self.n.ilog2() as usize + if self.n.is_power_of_two() { 0 } else { 1 }
    }
}

enum CoolingState {
    Additive {
        current: usize,
        target: usize,
    },
    Multiplicative {
        current: f64,
        factor: f64,
        times: usize,
        target: usize,
    },
    Infinite,
}

struct CoolingSchedule {
    config: CoolingConfig,
    state: CoolingState,
}

impl From<CoolingConfig> for CoolingSchedule {
    fn from(config: CoolingConfig) -> Self {
        let target = config.additive_ratio.get() * config.n.get() * config.log();
        CoolingSchedule {
            config,
            state: CoolingState::Additive { current: 0, target },
        }
    }
}

impl Iterator for CoolingSchedule {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let (value, state) = match self.state {
            CoolingState::Additive { current, target } => {
                let value = current as f64
                    / (self.config.n.get() * self.config.additive_ratio.get()) as f64;
                let state = if current == target {
                    let log = self.config.log();
                    let target =
                        log * log * self.config.n.get() * self.config.multiplicative_ratio.get();
                    let gamma = 1.0
                        + 1.0
                            / (self.config.n.get() * log * self.config.multiplicative_ratio.get())
                                as f64;
                    CoolingState::Multiplicative {
                        current: value * gamma,
                        factor: gamma,
                        times: 1,
                        target,
                    }
                } else {
                    CoolingState::Additive {
                        current: current + 1,
                        target,
                    }
                };
                (Some(value), state)
            }
            CoolingState::Multiplicative {
                current,
                factor,
                times,
                target,
            } => {
                let state = if times == target {
                    CoolingState::Infinite
                } else {
                    CoolingState::Multiplicative {
                        current: current * factor,
                        factor,
                        times: times + 1,
                        target,
                    }
                };
                (Some(current), state)
            }
            _ => (None, CoolingState::Infinite),
        };
        self.state = state;
        value
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn simple_cooling_schedule() {
        let config = CoolingConfig {
            n: NonZeroUsize::new(10).unwrap(),
            additive_ratio: NonZeroUsize::new(1).unwrap(),
            multiplicative_ratio: NonZeroUsize::new(1).unwrap(),
        };
        let mut schedule = CoolingSchedule::from(config);
        let mut last = schedule.next().unwrap();
        for value in schedule {
            assert!(value >= last);
            last = value;
            println!("{}", value)
        }
    }

    #[test]
    fn scaled_cooling_schedule() {
        let config = CoolingConfig {
            n: NonZeroUsize::new(16).unwrap(),
            additive_ratio: NonZeroUsize::new(4).unwrap(),
            multiplicative_ratio: NonZeroUsize::new(4).unwrap(),
        };
        let mut schedule = CoolingSchedule::from(config);
        let mut last = schedule.next().unwrap();
        for value in schedule {
            assert!(value >= last);
            last = value;
            println!("{}", value)
        }
    }
}
