"""The input tables the kernels are held to the reference on."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Series lengths: empty, shorter than every window, around the common windows, and long.
_LENGTHS = (0, 1, 2, 3, 13, 14, 15, 27, 50, 333)
#: Windows, periods and spans the kernels are called with — the graph's own and the edges.
_WINDOWS = (1, 2, 3, 5, 9, 14, 20, 26, 50)


def _series(seed: int, size: int) -> np.ndarray:
    """A finite float64 series; the seed picks one of eight shapes the TA-Lib ledger does not."""
    rng = np.random.default_rng(seed)
    shape = seed % 8
    series = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size)))
    if shape == 1:
        series *= 1e8
    elif shape == 2:
        series = rng.normal(0.0, 1.0, size) * 10.0 ** rng.integers(-6, 9, size).astype(float)
    elif shape == 3:
        series[size // 4 : size // 2] = 0.0
    elif shape == 4 and size:
        series[size // 3 :] = series[size // 3]
    elif shape == 5:
        series = np.round(series, 0)
    elif shape == 6:
        series = np.asarray(rng.choice(np.array([0.0, -0.0, 1.5, -1.5]), size), dtype=float)
    elif shape == 7:
        series = -series
    return series


def _gapped(seed: int, size: int) -> np.ndarray:
    """:func:`_series` with a leading ``NaN``, an interior gap and an infinity where there is room."""
    series = _series(seed, size).copy()
    if size > 4:
        series[0] = np.nan
        series[size // 2] = np.nan
        series[-1] = np.inf if seed % 2 else -np.inf
    return series


def _cases() -> Iterator[tuple[str, tuple[Any, ...]]]:
    """``(kernel name, arguments)`` over every kernel — each argument tuple used once, fresh."""
    for seed in range(16):
        for size in _LENGTHS:
            finite_run = _series(seed, size)
            gapped = _gapped(seed, size)
            other = np.abs(_series(seed + 100, size))
            flags = (np.arange(size) % 7 == 0).astype(float)
            if size > 5:
                flags[5] = np.nan
            for window in _WINDOWS:
                yield "ta_sum_kernel", (gapped, np.isfinite(gapped), window)
                yield "ta_var_kernel", (gapped, np.isfinite(gapped), window)
                yield "ema_smooth_run", (finite_run, window, 2.0 / (window + 1.0))
                yield "wilder_smooth_run", (finite_run, window)
                yield "wilder_rsi_smooth_run", (finite_run, window)
                yield "wilder_sum_run", (finite_run, window)
                for seed_offset in (window - 1, window, 2 * window + 3):
                    yield "ema_fma_at_run", (finite_run, window, seed_offset)
                yield "wilder_adx_smooth_run", (finite_run, window)
                if size >= window:
                    yield "cci_kernel", (finite_run, 0, window)
                money = np.abs(finite_run) * other
                yield (
                    "money_flow_index_kernel",
                    (np.isfinite(gapped), money * (np.arange(size) % 2), money * (np.arange(size) % 3 == 0), window),
                )
            yield "cumulative_within_session_kernel", (gapped, flags)
            yield "dx_kernel", (np.abs(gapped) % 50.0, other % 50.0)
            for fast, slow in ((3, 10), (2, 5), (1, 1)):
                first_output = max(fast, slow) - 1
                if first_output < size:
                    yield "chaikin_oscillator_kernel", (finite_run, 0, first_output, fast, slow)
            if size > 1:
                highs, lows = np.abs(finite_run) + other * 0.01, np.abs(finite_run) - other * 0.01
                yield "sar_run", (highs, lows, 0, size, np.full(size, np.nan))
