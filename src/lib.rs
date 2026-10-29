//! fnquant-kernels: TA-Lib-exact per-bar kernels.
//!
//! Each function is a statement-for-statement translation of a plain-Python loop that reproduces TA-Lib's
//! floating-point operations in TA-Lib's order (`tests/reference.py` holds those loops): the same IEEE-754 double
//! operations in the same order. Rust never contracts `a * b + c` into a fused multiply-add, and `f64::mul_add` is
//! the one explicit fused operation, standing where TA-Lib's C source calls `fma`. The algorithms are TA-Lib's
//! (BSD-3-Clause); this is an independent reimplementation, and TA-Lib is the oracle (`tests/test_talib.py`).
//!
//! **Python's semantics, not Rust's.** The kernels are held byte-for-byte to the Python loops on every case,
//! including inputs no caller should send (zero and negative periods), so the Python model is ported with them:
//! - every index goes through [`Col::at`] / [`Out::set`]: a negative index counts from the end, and one out of
//!   range raises `IndexError`;
//! - every `/` goes through [`div`]: a zero divisor raises `ZeroDivisionError`;
//! - `np.zeros(n)` with `n < 0` raises `ValueError` ([`zeros`]);
//! - `range(a, b)` is empty when `a >= b`.
//!
//! **A panic is an `Exception`.** `PyO3` raises a Rust panic as `PanicException`, a `BaseException`, which
//! `except Exception` does not catch. Every export runs its body under [`guarded`], which raises
//! `KernelPanic` (a `RuntimeError`) instead. No kernel is known to panic; `_panic` exists to test the path.

use std::panic::{AssertUnwindSafe, catch_unwind};

use numpy::ndarray::{ArrayView1, ArrayViewMut1};
use numpy::{PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::create_exception;
use pyo3::exceptions::{
    PyIndexError, PyMemoryError, PyOverflowError, PyRuntimeError, PyValueError, PyZeroDivisionError,
};
use pyo3::prelude::*;

create_exception!(
    fnquant_kernels,
    KernelPanic,
    PyRuntimeError,
    "A panic inside a kernel: a bug in the extension, never an input error."
);

type Out1<'py> = PyResult<Bound<'py, PyArray1<f64>>>;

/// `body()`, with a panic raised as `KernelPanic` rather than `PyO3`'s `PanicException`.
fn guarded<T>(body: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .map(|text| (*text).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "a kernel panicked".to_owned());
        Err(KernelPanic::new_err(message))
    })
}

/// Python's index rule: `-len <= index < len`, a negative one counting from the end.
fn position(index: i64, len: usize) -> PyResult<usize> {
    let len = len as i64;
    let wrapped = if index < 0 { index + len } else { index };
    if (0..len).contains(&wrapped) {
        Ok(wrapped as usize)
    } else {
        Err(PyIndexError::new_err(format!(
            "index {index} is out of bounds for axis 0 with size {len}"
        )))
    }
}

/// Python's float `/`: `ZeroDivisionError` on a zero (or negative-zero) divisor.
fn div(numerator: f64, divisor: f64) -> PyResult<f64> {
    if divisor == 0.0 {
        Err(PyZeroDivisionError::new_err("float division by zero"))
    } else {
        Ok(numerator / divisor)
    }
}

/// Integer arguments are held to ±2**53. Python's ints are unbounded, so beyond that the Python loops'
/// arithmetic cannot be matched in `i64`; refusing is the documented divergence (`OverflowError`, as for an int
/// that does not fit an `i64` at all), and it keeps every index and count computation below exact.
const INT_LIMIT: i64 = 1 << 53;

fn small(values: &[i64]) -> PyResult<()> {
    if values.iter().any(|value| value.unsigned_abs() > INT_LIMIT as u64) {
        return Err(PyOverflowError::new_err(
            "integer arguments must be within ±2**53",
        ));
    }
    Ok(())
}

/// `np.zeros(n)`.
fn zeros(n: i64) -> PyResult<Out> {
    if n < 0 {
        return Err(PyValueError::new_err("negative dimensions are not allowed"));
    }
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(n as usize)
        .map_err(|_| PyMemoryError::new_err("cannot allocate the kernel's buffer"))?;
    buffer.resize(n as usize, 0.0);
    Ok(Out(buffer))
}

/// Python's `max(a, b)` / `min(a, b)` on floats: the first argument unless the second strictly beats it.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}
fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// A read-only 1-D column, indexed the Python way; strided views need no copy.
struct Col<'a, T>(ArrayView1<'a, T>);

impl<T: Copy> Col<'_, T> {
    fn at(&self, index: i64) -> PyResult<T> {
        Ok(self.0[position(index, self.0.len())?])
    }
    fn size(&self) -> i64 {
        self.0.len() as i64
    }
}

/// A kernel's output buffer, written the Python way.
struct Out(Vec<f64>);

impl Out {
    fn nan(size: i64) -> Self {
        Out(vec![f64::NAN; size as usize])
    }
    fn at(&self, index: i64) -> PyResult<f64> {
        Ok(self.0[position(index, self.0.len())?])
    }
    fn set(&mut self, index: i64, value: f64) -> PyResult<()> {
        let at = position(index, self.0.len())?;
        self.0[at] = value;
        Ok(())
    }
    #[allow(clippy::unnecessary_wraps)] // every kernel's closure ends in this `PyResult`
    fn into_py(self, py: Python<'_>) -> Out1<'_> {
        Ok(PyArray1::from_vec(py, self.0))
    }
}

// ---------------------------------------------------------------- windows.py

#[pyfunction]
fn ta_sum_kernel<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<'py, f64>,
    finite: PyReadonlyArray1<'py, bool>,
    window: i64,
) -> Out1<'py> {
    guarded(|| {
        small(&[window])?;
        let (series, finite) = (Col(series.as_array()), Col(finite.as_array()));
        let size = series.size();
        let mut out = Out::nan(size);
        let mut i = 0;
        while i < size {
            while i < size && !finite.at(i)? {
                i += 1;
            }
            let run_start = i;
            let mut total = 0.0;
            let mut nonzero = 0_i64;
            let mut filled = 0_i64;
            while i < size && finite.at(i)? && filled < window - 1 {
                let value = series.at(i)?;
                total += value;
                if value != 0.0 {
                    nonzero += 1;
                }
                filled += 1;
                i += 1;
            }
            if filled < window - 1 {
                continue;
            }
            let mut trailing = run_start;
            while i < size && finite.at(i)? {
                let value = series.at(i)?;
                total += value;
                if value != 0.0 {
                    nonzero += 1;
                }
                if nonzero == 0 {
                    total = 0.0;
                }
                out.set(i, total)?;
                let leaving = series.at(trailing)?;
                total -= leaving;
                if leaving != 0.0 {
                    nonzero -= 1;
                }
                trailing += 1;
                i += 1;
            }
        }
        out.into_py(py)
    })
}

#[pyfunction]
fn ema_smooth_run<'py>(
    py: Python<'py>,
    run: PyReadonlyArray1<'py, f64>,
    periods: i64,
    alpha: f64,
) -> Out1<'py> {
    guarded(|| {
        small(&[periods])?;
        let run = Col(run.as_array());
        let mut out = Out::nan(run.size());
        if run.size() < periods {
            return out.into_py(py);
        }
        let mut total = 0.0;
        for i in 0..periods {
            total += run.at(i)?;
        }
        let mut previous = div(total, periods as f64)?;
        let seed_at = periods - 1;
        out.set(seed_at, previous)?;
        for i in seed_at + 1..run.size() {
            previous = (run.at(i)? - previous).mul_add(alpha, previous);
            out.set(i, previous)?;
        }
        out.into_py(py)
    })
}

#[pyfunction]
fn wilder_smooth_run<'py>(
    py: Python<'py>,
    run: PyReadonlyArray1<'py, f64>,
    periods: i64,
) -> Out1<'py> {
    guarded(|| {
        small(&[periods])?;
        let run = Col(run.as_array());
        let mut out = Out::nan(run.size());
        if run.size() < periods {
            return out.into_py(py);
        }
        let beta = div((periods - 1) as f64, periods as f64)?;
        let alpha = 1.0 - beta;
        let mut total = 0.0;
        for i in 0..periods {
            total += run.at(i)?;
        }
        let mut previous = div(total, periods as f64)?;
        let seed_at = periods - 1;
        out.set(seed_at, previous)?;
        for i in seed_at + 1..run.size() {
            previous = beta.mul_add(previous, alpha * run.at(i)?);
            out.set(i, previous)?;
        }
        out.into_py(py)
    })
}

#[pyfunction]
fn ta_var_kernel<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<'py, f64>,
    finite: PyReadonlyArray1<'py, bool>,
    window: i64,
) -> Out1<'py> {
    guarded(|| {
        small(&[window])?;
        let (series, finite) = (Col(series.as_array()), Col(finite.as_array()));
        let size = series.size();
        let mut out = Out::nan(size);
        let inverse = div(1.0, window as f64)?;
        let mut cursor = 0;
        while cursor < size {
            while cursor < size && !finite.at(cursor)? {
                cursor += 1;
            }
            let run_start = cursor;
            if size - run_start < window {
                break;
            }
            let mut trailing = run_start;
            let mut shift = series.at(trailing)?;
            let mut total1 = 0.0;
            let mut total2 = 0.0;
            let mut short_run = false;
            for j in run_start..run_start + window - 1 {
                if !finite.at(j)? {
                    short_run = true;
                    break;
                }
                let mut deviation = series.at(j)? - shift;
                total1 += deviation;
                deviation *= deviation;
                total2 += deviation;
            }
            if short_run {
                cursor = run_start + 1;
                continue;
            }
            let mut i = run_start + window - 1;
            let mut bars_since_reseed = 32 * window;
            while i < size && finite.at(i)? {
                let mut deviation = series.at(i)? - shift;
                total1 += deviation;
                deviation *= deviation;
                total2 += deviation;
                let mut mean1 = total1 * inverse;
                let mut variance = total2 * inverse - mean1 * mean1;
                deviation = series.at(trailing)? - shift;
                total1 -= deviation;
                deviation *= deviation;
                total2 -= deviation;
                trailing += 1;
                bars_since_reseed -= 1;
                if variance < 1e-6 * (total2 * inverse)
                    || deviation > 1e6 * total2
                    || bars_since_reseed <= 0
                {
                    bars_since_reseed = 32 * window;
                    let window_start = i - window + 1;
                    let mut running = 0.0;
                    for j in window_start..=i {
                        running += series.at(j)?;
                    }
                    shift = running * inverse;
                    total1 = 0.0;
                    total2 = 0.0;
                    for j in window_start..=i {
                        let mut deviation = series.at(j)? - shift;
                        total1 += deviation;
                        deviation *= deviation;
                        total2 += deviation;
                    }
                    mean1 = total1 * inverse;
                    variance = total2 * inverse - mean1 * mean1;
                    if variance < 1e-12 * (total2 * inverse) {
                        variance = 0.0;
                    }
                    deviation = series.at(window_start)? - shift;
                    total1 -= deviation;
                    deviation *= deviation;
                    total2 -= deviation;
                }
                out.set(i, variance)?;
                i += 1;
            }
            cursor = i;
        }
        out.into_py(py)
    })
}

#[pyfunction]
fn wilder_rsi_smooth_run<'py>(
    py: Python<'py>,
    run: PyReadonlyArray1<'py, f64>,
    periods: i64,
) -> Out1<'py> {
    guarded(|| {
        small(&[periods])?;
        let run = Col(run.as_array());
        let mut out = Out::nan(run.size());
        if run.size() < periods {
            return out.into_py(py);
        }
        let inverse = div(1.0, periods as f64)?;
        let mut total = 0.0;
        for i in 0..periods {
            total += run.at(i)?;
        }
        let mut previous = total * inverse;
        let seed_at = periods - 1;
        out.set(seed_at, previous)?;
        for i in seed_at + 1..run.size() {
            previous *= (periods - 1) as f64;
            previous += run.at(i)?;
            previous *= inverse;
            out.set(i, previous)?;
        }
        out.into_py(py)
    })
}

#[pyfunction]
fn wilder_sum_run<'py>(
    py: Python<'py>,
    run: PyReadonlyArray1<'py, f64>,
    periods: i64,
) -> Out1<'py> {
    guarded(|| {
        small(&[periods])?;
        let run = Col(run.as_array());
        let mut out = Out::nan(run.size());
        if run.size() < periods {
            return out.into_py(py);
        }
        let mut total = 0.0;
        for j in 0..periods - 1 {
            let stepped = total + run.at(j)?;
            total = py_max(stepped, total);
        }
        for i in periods - 1..run.size() {
            let decayed = total - div(total, periods as f64)?;
            total = decayed + run.at(i)?;
            total = py_max(total, decayed);
            out.set(i, total)?;
        }
        out.into_py(py)
    })
}

#[pyfunction]
fn ema_fma_at_run<'py>(
    py: Python<'py>,
    run: PyReadonlyArray1<'py, f64>,
    periods: i64,
    seed_offset: i64,
) -> Out1<'py> {
    guarded(|| {
        small(&[periods, seed_offset])?;
        let run = Col(run.as_array());
        let mut out = Out::nan(run.size());
        let seed_at = seed_offset;
        if seed_at >= run.size() || seed_offset - periods + 1 < 0 {
            return out.into_py(py);
        }
        let alpha = div(2.0, periods as f64 + 1.0)?;
        let mut total = 0.0;
        for i in seed_at - periods + 1..=seed_at {
            total += run.at(i)?;
        }
        let mut previous = div(total, periods as f64)?;
        out.set(seed_at, previous)?;
        for i in seed_at + 1..run.size() {
            previous = (run.at(i)? - previous).mul_add(alpha, previous);
            out.set(i, previous)?;
        }
        out.into_py(py)
    })
}

#[pyfunction]
fn wilder_adx_smooth_run<'py>(
    py: Python<'py>,
    run: PyReadonlyArray1<'py, f64>,
    periods: i64,
) -> Out1<'py> {
    guarded(|| {
        small(&[periods])?;
        let run = Col(run.as_array());
        let mut out = Out::nan(run.size());
        if run.size() < periods {
            return out.into_py(py);
        }
        let mut total = 0.0;
        for i in 0..periods {
            total += run.at(i)?;
        }
        let mut previous = div(total, periods as f64)?;
        let seed_at = periods - 1;
        out.set(seed_at, previous)?;
        for i in seed_at + 1..run.size() {
            previous = div(previous * (periods - 1) as f64 + run.at(i)?, periods as f64)?;
            out.set(i, previous)?;
        }
        out.into_py(py)
    })
}

#[pyfunction]
fn cumulative_within_session_kernel<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<'py, f64>,
    resets: PyReadonlyArray1<'py, f64>,
) -> Out1<'py> {
    guarded(|| {
        let (series, resets) = (Col(series.as_array()), Col(resets.as_array()));
        let mut out = Out(vec![0.0; series.size() as usize]);
        let mut running = f64::NAN;
        for i in 0..series.size() {
            let flag = resets.at(i)?;
            if flag == 1.0 {
                running = 0.0;
            } else if flag.is_nan() {
                running = f64::NAN;
            }
            running += series.at(i)?;
            out.set(i, running)?;
        }
        out.into_py(py)
    })
}

// ------------------------------------------------------------- indicators.py

/// TA-Lib's `TA_IS_ZERO` bound.
const TA_EPSILON: f64 = 1e-14;

#[pyfunction]
fn cci_kernel<'py>(
    py: Python<'py>,
    typical_price: PyReadonlyArray1<'py, f64>,
    start: i64,
    period: i64,
) -> Out1<'py> {
    guarded(|| {
        small(&[start, period])?;
        let typical_price = Col(typical_price.as_array());
        let mut out = Out::nan(typical_price.size());
        let mut buffer = zeros(period)?;
        let mut position = 0;
        for i in start..start + period - 1 {
            buffer.set(position, typical_price.at(i)?)?;
            position = if position >= period - 1 {
                0
            } else {
                position + 1
            };
        }
        for i in start + period - 1..typical_price.size() {
            let latest = typical_price.at(i)?;
            buffer.set(position, latest)?;
            let mut average = 0.0;
            for j in 0..period {
                average += buffer.at(j)?;
            }
            average = div(average, period as f64)?;
            let mut deviation = 0.0;
            for j in 0..period {
                deviation += (buffer.at(j)? - average).abs();
            }
            deviation = div(deviation, period as f64)?;
            let numerator = latest - average;
            let magnitude = average.abs();
            let scaled = 0.015 * deviation;
            let negligible = numerator.abs() <= TA_EPSILON * magnitude
                || deviation.abs() <= TA_EPSILON * magnitude
                || scaled == 0.0;
            out.set(
                i,
                if negligible {
                    0.0
                } else {
                    div(numerator, scaled)?
                },
            )?;
            position = if position >= period - 1 {
                0
            } else {
                position + 1
            };
        }
        out.into_py(py)
    })
}

#[pyfunction]
fn dx_kernel<'py>(
    py: Python<'py>,
    plus_di: PyReadonlyArray1<'py, f64>,
    minus_di: PyReadonlyArray1<'py, f64>,
) -> Out1<'py> {
    guarded(|| {
        let (plus_di, minus_di) = (Col(plus_di.as_array()), Col(minus_di.as_array()));
        let mut out = Out::nan(plus_di.size());
        let mut previous = 0.0;
        for i in 0..plus_di.size() {
            let (plus, minus) = (plus_di.at(i)?, minus_di.at(i)?);
            if !(plus.is_finite() && minus.is_finite()) {
                previous = 0.0;
                continue;
            }
            let total = minus + plus;
            // numpy scalars: a zero `total` cannot pass the epsilon test, so this `/` never meets one.
            if total.abs() >= TA_EPSILON {
                previous = 100.0 * ((minus - plus).abs() / total);
            }
            out.set(i, previous)?;
        }
        out.into_py(py)
    })
}

#[pyfunction]
fn money_flow_index_kernel<'py>(
    py: Python<'py>,
    finite: PyReadonlyArray1<'py, bool>,
    positive_money_flow: PyReadonlyArray1<'py, f64>,
    negative_money_flow: PyReadonlyArray1<'py, f64>,
    period: i64,
) -> Out1<'py> {
    guarded(|| {
        small(&[period])?;
        let finite = Col(finite.as_array());
        let (positive_money_flow, negative_money_flow) = (
            Col(positive_money_flow.as_array()),
            Col(negative_money_flow.as_array()),
        );
        let size = finite.size();
        let mut out = Out::nan(size);
        let mut cursor = 0;
        while cursor < size {
            while cursor < size && !finite.at(cursor)? {
                cursor += 1;
            }
            if size - cursor < period + 1 {
                break;
            }
            let mut positive = zeros(period)?;
            let mut negative = zeros(period)?;
            let mut slot = 0;
            let mut positive_sum = 0.0;
            let mut negative_sum = 0.0;
            let mut null_run = 0;
            let mut step = 0;
            let mut today = cursor + 1;
            while today < size && finite.at(today)? {
                if step >= period {
                    positive_sum -= positive.at(slot)?;
                    negative_sum -= negative.at(slot)?;
                }
                positive.set(slot, positive_money_flow.at(today)?)?;
                negative.set(slot, negative_money_flow.at(today)?)?;
                positive_sum += positive.at(slot)?;
                negative_sum += negative.at(slot)?;
                null_run = if positive.at(slot)? == 0.0 && negative.at(slot)? == 0.0 {
                    null_run + 1
                } else {
                    0
                };
                if null_run >= period {
                    null_run = period;
                    positive_sum = 0.0;
                    negative_sum = 0.0;
                }
                if step >= period - 1 {
                    let total = positive_sum + negative_sum;
                    let clamped = if positive_sum < 0.0 {
                        0.0
                    } else {
                        py_min(positive_sum, total)
                    };
                    out.set(
                        today,
                        if total <= 0.0 {
                            0.0
                        } else {
                            100.0 * div(clamped, total)?
                        },
                    )?;
                }
                slot = if slot >= period - 1 { 0 } else { slot + 1 };
                step += 1;
                today += 1;
            }
            cursor = today;
        }
        out.into_py(py)
    })
}

#[pyfunction]
fn chaikin_oscillator_kernel<'py>(
    py: Python<'py>,
    accumulation_distribution: PyReadonlyArray1<'py, f64>,
    start: i64,
    first_output: i64,
    fast_span: i64,
    slow_span: i64,
) -> Out1<'py> {
    guarded(|| {
        small(&[start, first_output, fast_span, slow_span])?;
        let accumulation_distribution = Col(accumulation_distribution.as_array());
        let mut out = Out::nan(accumulation_distribution.size());
        let fast_weight = div(2.0, fast_span as f64 + 1.0)?;
        let fast_keep = 1.0 - fast_weight;
        let slow_weight = div(2.0, slow_span as f64 + 1.0)?;
        let slow_keep = 1.0 - slow_weight;
        let mut fast = accumulation_distribution.at(start)?;
        let mut slow = fast;
        for i in start + 1..accumulation_distribution.size() {
            let level = accumulation_distribution.at(i)?;
            fast = fast_keep.mul_add(fast, fast_weight * level);
            slow = slow_keep.mul_add(slow, slow_weight * level);
            if i >= first_output {
                out.set(i, fast - slow)?;
            }
        }
        out.into_py(py)
    })
}

/// Python's `max(a, b, c)` / `min(a, b, c)`.
fn py_max3(a: f64, b: f64, c: f64) -> f64 {
    py_max(py_max(a, b), c)
}
fn py_min3(a: f64, b: f64, c: f64) -> f64 {
    py_min(py_min(a, b), c)
}

/// Writes into `out` and returns that same array, as the Python did.
fn sar_into(
    highs: &Col<'_, f64>,
    lows: &Col<'_, f64>,
    first: i64,
    stop: i64,
    out: &mut ArrayViewMut1<'_, f64>,
) -> PyResult<()> {
    let acceleration = 0.02_f64;
    let maximum = 0.2;
    let up_move = highs.at(first + 1)? - highs.at(first)?;
    let down_move = lows.at(first)? - lows.at(first + 1)?;
    let seed_minus_dm = if down_move > 0.0 && up_move < down_move {
        down_move
    } else {
        0.0
    };
    // `not x > 0.0`, as the Python wrote it: a NaN seed is long, which `x <= 0.0` would get wrong.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    let mut is_long = !(seed_minus_dm > 0.0);
    let (mut extreme, mut sar) = if is_long {
        (highs.at(first + 1)?, lows.at(first)?)
    } else {
        (lows.at(first + 1)?, highs.at(first)?)
    };
    let mut af = acceleration;
    let mut prior_high = highs.at(first + 1)?;
    let mut prior_low = lows.at(first + 1)?;
    let size = out.len();
    let mut write = |index: i64, value: f64| -> PyResult<()> {
        out[position(index, size)?] = value;
        Ok(())
    };
    for i in first + 1..stop {
        let high_i = highs.at(i)?;
        let low_i = lows.at(i)?;
        if is_long {
            if low_i <= sar {
                is_long = false;
                let flipped = py_max3(extreme, prior_high, high_i);
                write(i, flipped)?;
                af = acceleration;
                extreme = low_i;
                sar = af.mul_add(extreme - flipped, flipped);
                sar = py_max3(sar, prior_high, high_i);
            } else {
                write(i, sar)?;
                if high_i > extreme {
                    extreme = high_i;
                    af = py_min(af + acceleration, maximum);
                }
                sar = af.mul_add(extreme - sar, sar);
                sar = py_min3(sar, prior_low, low_i);
            }
        } else if high_i >= sar {
            is_long = true;
            let flipped = py_min3(extreme, prior_low, low_i);
            write(i, flipped)?;
            af = acceleration;
            extreme = high_i;
            sar = af.mul_add(extreme - flipped, flipped);
            sar = py_min3(sar, prior_low, low_i);
        } else {
            write(i, sar)?;
            if low_i < extreme {
                extreme = low_i;
                af = py_min(af + acceleration, maximum);
            }
            sar = af.mul_add(extreme - sar, sar);
            sar = py_max3(sar, prior_high, high_i);
        }
        prior_high = high_i;
        prior_low = low_i;
    }
    Ok(())
}

#[pyfunction]
fn sar_run<'py>(
    highs: PyReadonlyArray1<'py, f64>,
    lows: PyReadonlyArray1<'py, f64>,
    first: i64,
    stop: i64,
    mut out: PyReadwriteArray1<'py, f64>,
) -> Out1<'py> {
    guarded(|| {
        small(&[first, stop])?;
        let (highs, lows) = (Col(highs.as_array()), Col(lows.as_array()));
        let target: Bound<'py, PyArray1<f64>> = (**out).clone();
        sar_into(&highs, &lows, first, stop, &mut out.as_array_mut())?;
        Ok(target)
    })
}

/// Panics, so a test can show the panic reaches Python as `KernelPanic`.
#[pyfunction]
fn _panic() -> PyResult<()> {
    guarded(|| panic!("deliberate panic from fnquant_kernels._panic"))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("KernelPanic", m.py().get_type::<KernelPanic>())?;
    m.add_function(wrap_pyfunction!(ta_sum_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(ema_smooth_run, m)?)?;
    m.add_function(wrap_pyfunction!(wilder_smooth_run, m)?)?;
    m.add_function(wrap_pyfunction!(ta_var_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(wilder_rsi_smooth_run, m)?)?;
    m.add_function(wrap_pyfunction!(wilder_sum_run, m)?)?;
    m.add_function(wrap_pyfunction!(ema_fma_at_run, m)?)?;
    m.add_function(wrap_pyfunction!(wilder_adx_smooth_run, m)?)?;
    m.add_function(wrap_pyfunction!(cumulative_within_session_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(cci_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(dx_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(money_flow_index_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(chaikin_oscillator_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(sar_run, m)?)?;
    m.add_function(wrap_pyfunction!(_panic, m)?)
}
