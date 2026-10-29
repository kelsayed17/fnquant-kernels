# fnquant-kernels

TA-Lib-exact per-bar numeric kernels for Python, compiled from Rust.

TA-Lib's recursive indicators (EMA, Wilder smoothing, SAR, CCI, MFI, rolling variance, …) are bit-exact only when
the floating-point operations run in TA-Lib's own order; no `numpy` reduction reproduces that. These are 14 such
loops as a [PyO3](https://pyo3.rs) extension, each a statement-for-statement translation of a plain-Python
reference (`tests/reference.py`):

- the same IEEE-754 double operations in the same order, no fast-math, no implicit fused multiply-add;
  `f64::mul_add` stands exactly where TA-Lib's C source calls `fma`;
- Python's semantics at the edges: negative indices wrap, an out-of-range index raises `IndexError`, a zero
  divisor raises `ZeroDivisionError`;
- `#![forbid(unsafe_code)]`; a Rust panic raises `fnquant_kernels.KernelPanic`, a `RuntimeError`, never a
  `BaseException`;
- about 0.7 MB installed, about a millisecond for the first call, no JIT, no cache directory.

```python
import numpy as np
from fnquant_kernels import ema_smooth_run

ema_smooth_run(np.linspace(1.0, 9.0, 9), 3, 2.0 / (3 + 1.0))
```

The kernels take float64 arrays (and boolean masks where they read a finite mask) and return float64 arrays.
They are low-level: each expects its caller to have split the input into finite runs, as the reference does.

## Provenance and license

The algorithms are those of [TA-Lib](https://ta-lib.org) (BSD-3-Clause). This is an independent reimplementation
and contains no TA-Lib source; TA-Lib is the test oracle (`tests/test_talib.py`). Apache-2.0, see `LICENSE`.

## Development

```bash
uv sync
uv run maturin develop --release
bash scripts/check.sh
```
