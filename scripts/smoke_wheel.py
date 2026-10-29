"""Smoke test run against an installed wheel: import it, run a kernel, and see a panic arrive as an Exception."""

import fnquant_kernels
import numpy as np
from fnquant_kernels import _native

series = np.linspace(1.0, 9.0, 9)
ema = fnquant_kernels.ema_smooth_run(series, 3, 0.5)
assert ema.shape == series.shape
assert ema[2] == 2.0  # the SMA seed of 1, 2, 3

try:
    _native._panic()  # noqa: SLF001
except Exception as error:  # noqa: BLE001
    assert isinstance(error, fnquant_kernels.KernelPanic), error
else:
    msg = "a panic did not raise"
    raise AssertionError(msg)
print("smoke ok")
