"""The kernels against TA-Lib itself, the oracle the reference loops were written to reproduce."""

from __future__ import annotations

import fnquant_kernels
import numpy as np
import pytest
import talib


def _closes(seed: int, size: int = 400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size)))


@pytest.mark.parametrize("period", [2, 3, 5, 12, 26, 50])
@pytest.mark.parametrize("seed", range(4))
def test_ema_is_bit_exact_with_talib_ema(period: int, seed: int) -> None:
    closes = _closes(seed)
    ours = fnquant_kernels.ema_smooth_run(closes, period, 2.0 / (period + 1.0))
    assert ours.tobytes() == talib.EMA(closes, period).tobytes()


@pytest.mark.parametrize("window", [2, 5, 14, 20, 50])
@pytest.mark.parametrize("seed", range(4))
def test_variance_is_bit_exact_with_talib_var(window: int, seed: int) -> None:
    closes = _closes(seed)
    ours = fnquant_kernels.ta_var_kernel(closes, np.isfinite(closes), window)
    assert ours.tobytes() == talib.VAR(closes, window, 1).tobytes()


@pytest.mark.parametrize("period", [5, 14, 20])
@pytest.mark.parametrize("seed", range(4))
def test_cci_is_bit_exact_with_talib_cci(period: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    closes = _closes(seed)
    highs = closes * (1.0 + np.abs(rng.normal(0.0, 0.01, closes.size)))
    lows = closes * (1.0 - np.abs(rng.normal(0.0, 0.01, closes.size)))
    typical = (highs + lows + closes) / 3.0
    ours = fnquant_kernels.cci_kernel(typical, 0, period)
    assert ours.tobytes() == talib.CCI(highs, lows, closes, period).tobytes()
