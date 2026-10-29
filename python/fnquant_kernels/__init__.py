"""TA-Lib-exact per-bar kernels, compiled from Rust.

Each kernel answers byte-for-byte what the plain-Python loop it translates answers (``tests/reference.py``), and
raises the same exception type where that loop raises. A panic inside the extension raises :class:`KernelPanic`,
an ``Exception``.
"""

from fnquant_kernels._native import (
    KernelPanic,
    cci_kernel,
    chaikin_oscillator_kernel,
    cumulative_within_session_kernel,
    dx_kernel,
    ema_fma_at_run,
    ema_smooth_run,
    money_flow_index_kernel,
    sar_run,
    ta_sum_kernel,
    ta_var_kernel,
    wilder_adx_smooth_run,
    wilder_rsi_smooth_run,
    wilder_smooth_run,
    wilder_sum_run,
)

__all__ = [
    "KernelPanic",
    "cci_kernel",
    "chaikin_oscillator_kernel",
    "cumulative_within_session_kernel",
    "dx_kernel",
    "ema_fma_at_run",
    "ema_smooth_run",
    "money_flow_index_kernel",
    "sar_run",
    "ta_sum_kernel",
    "ta_var_kernel",
    "wilder_adx_smooth_run",
    "wilder_rsi_smooth_run",
    "wilder_smooth_run",
    "wilder_sum_run",
]
