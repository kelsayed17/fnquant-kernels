"""The pure-Python reference loops the compiled kernels are held to, byte for byte.

Each function is the plain-Python statement sequence a kernel translates: the same IEEE-754 double operations
in the same order, ``math.fma`` where TA-Lib's C source calls ``fma``. They are slow and exist only as the oracle
for ``test_parity.py``; ``test_talib.py`` ties the same loops to TA-Lib itself.
"""

from __future__ import annotations

from math import fma, isnan

import numpy as np

_TA_EPSILON = 1e-14


def ta_sum_kernel(series: np.ndarray, finite: np.ndarray, window: int) -> np.ndarray:
    size = series.size
    out = np.full(size, np.nan)
    i = 0
    while i < size:
        while i < size and (not finite[i]):
            i += 1
        run_start = i
        total = 0.0
        nonzero = 0
        filled = 0
        while i < size and finite[i] and (filled < window - 1):
            value = float(series[i])
            total += value
            if value != 0.0:
                nonzero += 1
            filled += 1
            i += 1
        if filled < window - 1:
            continue
        trailing = run_start
        while i < size and finite[i]:
            value = float(series[i])
            total += value
            if value != 0.0:
                nonzero += 1
            if nonzero == 0:
                total = 0.0
            out[i] = total
            leaving = float(series[trailing])
            total -= leaving
            if leaving != 0.0:
                nonzero -= 1
            trailing += 1
            i += 1
    return out


def ema_smooth_run(run: np.ndarray, periods: int, alpha: float) -> np.ndarray:
    out = np.full(run.size, np.nan)
    if run.size < periods:
        return out
    total = 0.0
    for i in range(periods):
        total += float(run[i])
    previous = total / periods
    seed_at = periods - 1
    out[seed_at] = previous
    for i in range(seed_at + 1, run.size):
        previous = fma(float(run[i]) - previous, alpha, previous)
        out[i] = previous
    return out


def wilder_smooth_run(run: np.ndarray, periods: int) -> np.ndarray:
    out = np.full(run.size, np.nan)
    if run.size < periods:
        return out
    beta = (periods - 1) / periods
    alpha = 1.0 - beta
    total = 0.0
    for i in range(periods):
        total += float(run[i])
    previous = total / periods
    seed_at = periods - 1
    out[seed_at] = previous
    for i in range(seed_at + 1, run.size):
        previous = fma(beta, previous, alpha * float(run[i]))
        out[i] = previous
    return out


def ta_var_kernel(series: np.ndarray, finite: np.ndarray, window: int) -> np.ndarray:
    out = np.full(series.size, np.nan)
    size = series.size
    inverse = 1.0 / window
    cursor = 0
    while cursor < size:
        while cursor < size and (not finite[cursor]):
            cursor += 1
        run_start = cursor
        if size - run_start < window:
            break
        trailing = run_start
        shift = float(series[trailing])
        total1 = 0.0
        total2 = 0.0
        short_run = False
        for j in range(run_start, run_start + window - 1):
            if not finite[j]:
                short_run = True
                break
            deviation = float(series[j]) - shift
            total1 += deviation
            deviation *= deviation
            total2 += deviation
        if short_run:
            cursor = run_start + 1
            continue
        i = run_start + window - 1
        bars_since_reseed = 32 * window
        while i < size and finite[i]:
            deviation = float(series[i]) - shift
            total1 += deviation
            deviation *= deviation
            total2 += deviation
            mean1 = total1 * inverse
            variance = total2 * inverse - mean1 * mean1
            deviation = float(series[trailing]) - shift
            total1 -= deviation
            deviation *= deviation
            total2 -= deviation
            trailing += 1
            bars_since_reseed -= 1
            if variance < 1e-06 * (total2 * inverse) or deviation > 1000000.0 * total2 or bars_since_reseed <= 0:
                bars_since_reseed = 32 * window
                window_start = i - window + 1
                running = 0.0
                for j in range(window_start, i + 1):
                    running += float(series[j])
                shift = running * inverse
                total1 = 0.0
                total2 = 0.0
                for j in range(window_start, i + 1):
                    deviation = float(series[j]) - shift
                    total1 += deviation
                    deviation *= deviation
                    total2 += deviation
                mean1 = total1 * inverse
                variance = total2 * inverse - mean1 * mean1
                if variance < 1e-12 * (total2 * inverse):
                    variance = 0.0
                deviation = float(series[window_start]) - shift
                total1 -= deviation
                deviation *= deviation
                total2 -= deviation
            out[i] = variance
            i += 1
        cursor = i
    return out


def wilder_rsi_smooth_run(run: np.ndarray, periods: int) -> np.ndarray:
    out = np.full(run.size, np.nan)
    if run.size < periods:
        return out
    inverse = 1.0 / periods
    total = 0.0
    for i in range(periods):
        total += float(run[i])
    previous = total * inverse
    seed_at = periods - 1
    out[seed_at] = previous
    for i in range(seed_at + 1, run.size):
        previous *= periods - 1
        previous += float(run[i])
        previous *= inverse
        out[i] = previous
    return out


def wilder_sum_run(run: np.ndarray, periods: int) -> np.ndarray:
    out = np.full(run.size, np.nan)
    if run.size < periods:
        return out
    total = 0.0
    for j in range(periods - 1):
        stepped = total + float(run[j])
        total = max(stepped, total)
    for i in range(periods - 1, run.size):
        decayed = total - total / periods
        total = decayed + float(run[i])
        total = max(total, decayed)
        out[i] = total
    return out


def ema_fma_at_run(run: np.ndarray, periods: int, seed_offset: int) -> np.ndarray:
    out = np.full(run.size, np.nan)
    seed_at = seed_offset
    if seed_at >= run.size or seed_offset - periods + 1 < 0:
        return out
    alpha = 2.0 / (periods + 1.0)
    total = 0.0
    for i in range(seed_at - periods + 1, seed_at + 1):
        total += float(run[i])
    previous = total / periods
    out[seed_at] = previous
    for i in range(seed_at + 1, run.size):
        previous = fma(float(run[i]) - previous, alpha, previous)
        out[i] = previous
    return out


def wilder_adx_smooth_run(run: np.ndarray, periods: int) -> np.ndarray:
    out = np.full(run.size, np.nan)
    if run.size < periods:
        return out
    total = 0.0
    for i in range(periods):
        total += float(run[i])
    previous = total / periods
    seed_at = periods - 1
    out[seed_at] = previous
    for i in range(seed_at + 1, run.size):
        previous = (previous * (periods - 1) + float(run[i])) / periods
        out[i] = previous
    return out


def cumulative_within_session_kernel(series: np.ndarray, resets: np.ndarray) -> np.ndarray:
    out = np.empty(series.size)
    running = np.nan
    for i in range(series.size):
        flag = resets[i]
        if flag == 1.0:
            running = 0.0
        elif isnan(flag):
            running = np.nan
        running += series[i]
        out[i] = running
    return out


def cci_kernel(typical_price: np.ndarray, start: int, period: int) -> np.ndarray:
    out = np.full(typical_price.size, np.nan)
    buffer = np.zeros(period)
    position = 0
    for i in range(start, start + period - 1):
        buffer[position] = float(typical_price[i])
        position = 0 if position >= period - 1 else position + 1
    for i in range(start + period - 1, typical_price.size):
        latest = float(typical_price[i])
        buffer[position] = latest
        average = 0.0
        for j in range(period):
            average += buffer[j]
        average /= period
        deviation = 0.0
        for j in range(period):
            deviation += abs(buffer[j] - average)
        deviation /= period
        numerator = latest - average
        magnitude = abs(average)
        scaled = 0.015 * deviation
        negligible = (
            abs(numerator) <= _TA_EPSILON * magnitude or abs(deviation) <= _TA_EPSILON * magnitude or scaled == 0.0
        )
        out[i] = 0.0 if negligible else numerator / scaled
        position = 0 if position >= period - 1 else position + 1
    return out


def dx_kernel(plus_di: np.ndarray, minus_di: np.ndarray) -> np.ndarray:
    out = np.full(plus_di.size, np.nan)
    previous = 0.0
    for i in range(plus_di.size):
        if not (np.isfinite(plus_di[i]) and np.isfinite(minus_di[i])):
            previous = 0.0
            continue
        total = minus_di[i] + plus_di[i]
        if abs(total) >= _TA_EPSILON:
            previous = 100.0 * (abs(minus_di[i] - plus_di[i]) / total)
        out[i] = previous
    return out


def money_flow_index_kernel(
    finite: np.ndarray, positive_money_flow: np.ndarray, negative_money_flow: np.ndarray, period: int
) -> np.ndarray:
    size = finite.size
    out = np.full(size, np.nan)
    cursor = 0
    while cursor < size:
        while cursor < size and (not finite[cursor]):
            cursor += 1
        if size - cursor < period + 1:
            break
        positive = np.zeros(period)
        negative = np.zeros(period)
        slot = 0
        positive_sum = 0.0
        negative_sum = 0.0
        null_run = 0
        step = 0
        today = cursor + 1
        while today < size and finite[today]:
            if step >= period:
                positive_sum -= positive[slot]
                negative_sum -= negative[slot]
            positive[slot] = float(positive_money_flow[today])
            negative[slot] = float(negative_money_flow[today])
            positive_sum += positive[slot]
            negative_sum += negative[slot]
            null_run = null_run + 1 if positive[slot] == 0.0 and negative[slot] == 0.0 else 0
            if null_run >= period:
                null_run = period
                positive_sum = 0.0
                negative_sum = 0.0
            if step >= period - 1:
                total = positive_sum + negative_sum
                clamped = 0.0 if positive_sum < 0.0 else min(positive_sum, total)
                out[today] = 0.0 if total <= 0.0 else 100.0 * (clamped / total)
            slot = 0 if slot >= period - 1 else slot + 1
            step += 1
            today += 1
        cursor = today
    return out


def chaikin_oscillator_kernel(
    accumulation_distribution: np.ndarray, start: int, first_output: int, fast_span: int, slow_span: int
) -> np.ndarray:
    out = np.full(accumulation_distribution.size, np.nan)
    fast_weight = 2.0 / (fast_span + 1.0)
    fast_keep = 1.0 - fast_weight
    slow_weight = 2.0 / (slow_span + 1.0)
    slow_keep = 1.0 - slow_weight
    fast = slow = float(accumulation_distribution[start])
    for i in range(start + 1, accumulation_distribution.size):
        level = float(accumulation_distribution[i])
        fast = fma(fast_keep, fast, fast_weight * level)
        slow = fma(slow_keep, slow, slow_weight * level)
        if i >= first_output:
            out[i] = fast - slow
    return out


def sar_run(highs: np.ndarray, lows: np.ndarray, first: int, stop: int, out: np.ndarray) -> np.ndarray:
    acceleration = 0.02
    maximum = 0.2
    up_move = highs[first + 1] - highs[first]
    down_move = lows[first] - lows[first + 1]
    seed_minus_dm = down_move if down_move > 0.0 and up_move < down_move else 0.0
    is_long = not seed_minus_dm > 0.0
    if is_long:
        extreme = highs[first + 1]
        sar = lows[first]
    else:
        extreme = lows[first + 1]
        sar = highs[first]
    af = acceleration
    prior_high = highs[first + 1]
    prior_low = lows[first + 1]
    for i in range(first + 1, stop):
        high_i = highs[i]
        low_i = lows[i]
        if is_long:
            if low_i <= sar:
                is_long = False
                flipped = max(extreme, prior_high, high_i)
                out[i] = flipped
                af = acceleration
                extreme = low_i
                sar = fma(af, extreme - flipped, flipped)
                sar = max(sar, prior_high, high_i)
            else:
                out[i] = sar
                if high_i > extreme:
                    extreme = high_i
                    af = min(af + acceleration, maximum)
                sar = fma(af, extreme - sar, sar)
                sar = min(sar, prior_low, low_i)
        elif high_i >= sar:
            is_long = True
            flipped = min(extreme, prior_low, low_i)
            out[i] = flipped
            af = acceleration
            extreme = high_i
            sar = fma(af, extreme - flipped, flipped)
            sar = min(sar, prior_low, low_i)
        else:
            out[i] = sar
            if low_i < extreme:
                extreme = low_i
                af = min(af + acceleration, maximum)
            sar = fma(af, extreme - sar, sar)
            sar = max(sar, prior_high, high_i)
        prior_high = high_i
        prior_low = low_i
    return out
