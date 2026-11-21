//! Small numeric helpers: ln n!, the normal quantile, log-sum-exp.

/// ln(n!) by direct summation (exact to rounding for the sizes used here).
pub fn ln_factorial(n: usize) -> f64 {
    (2..=n).map(|k| (k as f64).ln()).sum()
}

pub fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    max + values.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
}

/// Inverse standard normal CDF (Acklam's rational approximation, relative
/// error below 1.2e-9 over the open unit interval).
pub fn normal_quantile(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "quantile argument out of range: {p}");
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.38357751867269e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let low = 0.02425;
    if p < low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - low {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -normal_quantile(1.0 - p)
    }
}

/// Format exp(ln_value) as mantissa x 10^exponent without overflow.
pub fn scientific(ln_value: f64) -> String {
    if ln_value == f64::NEG_INFINITY {
        return "0".into();
    }
    let log10 = ln_value / std::f64::consts::LN_10;
    let exponent = log10.floor();
    format!("{:.6}e{:+}", 10f64.powf(log10 - exponent), exponent as i64)
}

/// SplitMix64 finalizer, used to derive independent per-sample seeds from
/// (run seed, phase, level, index) so results do not depend on scheduling.
pub fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn stream_seed(seed: u64, phase: u64, level: u64, index: u64) -> u64 {
    mix(mix(mix(mix(seed) ^ phase) ^ level) ^ index)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn quantiles_match_known_values() {
        assert!((normal_quantile(0.975) - 1.959964).abs() < 1e-5);
        assert!((normal_quantile(0.5)).abs() < 1e-9);
        assert!((normal_quantile(0.001) + 3.090232).abs() < 1e-5);
    }

    #[test]
    fn factorial_and_lse() {
        assert!((ln_factorial(10) - 3628800f64.ln()).abs() < 1e-12);
        let v = [0.0f64.ln(), 1.0f64.ln(), 3.0f64.ln()];
        assert!((log_sum_exp(&v) - 4.0f64.ln()).abs() < 1e-12);
    }
}
