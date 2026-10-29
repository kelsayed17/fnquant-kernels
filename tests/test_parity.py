"""fnquant_kernels — every kernel answers what its plain-Python reference answers, byte for byte.

``tests/reference.py`` holds the reference loops. This suite holds the extension to them on the ``cases`` tables
(gaps, infinities, signed zeros, 1e8 magnitudes, lengths 0 to 333, windows 1 to 50) and on a probe table of
inputs no caller should send, zero and negative periods, windows and spans and out-of-range starts, where the
outcome compared is the output bytes or the exception TYPE.
"""

from __future__ import annotations

import contextlib
import itertools
from typing import TYPE_CHECKING, Any

import fnquant_kernels
import numpy as np
import pytest
from fnquant_kernels import _native

from tests import reference
from tests.cases import _cases

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_REFERENCE: dict[str, Callable[..., np.ndarray]] = {
    name: function
    for name, function in vars(reference).items()
    if name.endswith(("kernel", "run")) and callable(function)
}
_COMPILED: dict[str, Callable[..., np.ndarray]] = {name: getattr(fnquant_kernels, name) for name in _REFERENCE}


def _bits(value: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    return value.dtype.str, tuple(int(axis) for axis in value.shape), value.tobytes()


def _python(name: str) -> Callable[..., np.ndarray]:
    return _REFERENCE[name]


def _fresh(arguments: tuple[Any, ...]) -> tuple[Any, ...]:
    # Each side gets its own copies: ``sar_run`` writes into its ``out`` argument.
    return tuple(a.copy() if isinstance(a, np.ndarray) else a for a in arguments)


def _outcome(function: Callable[..., np.ndarray], arguments: tuple[Any, ...]) -> object:
    try:
        return _bits(function(*_fresh(arguments)))
    except Exception as error:  # noqa: BLE001 — the exception's type IS the outcome compared
        return type(error)


def test_the_extension_exports_every_kernel_and_nothing_else() -> None:
    assert len(_REFERENCE) == 14
    exported = {name for name in fnquant_kernels.__all__ if name != "KernelPanic"}
    assert exported == set(_REFERENCE)


@pytest.mark.parametrize("name", sorted(_REFERENCE))
def test_the_extension_is_the_python_kernel_byte_for_byte(name: str) -> None:
    checked = 0
    for case_name, arguments in _cases():
        if case_name != name:
            continue
        expected = _python(name)(*_fresh(arguments))
        produced = _COMPILED[name](*_fresh(arguments))
        assert _bits(produced) == _bits(expected), f"{name}{tuple(getattr(a, 'shape', a) for a in arguments)}"
        checked += 1
    assert checked


#: Period-like arguments no node sends: zero and negative, plus 1 as the control.
_ODD = (-7, -3, -2, -1, 0, 1)
_PROBE_SIZES = (0, 1, 2, 3, 5, 9)


def _probes() -> Iterator[tuple[str, tuple[Any, ...]]]:
    """``(kernel name, arguments)`` for the probe table: every odd period on every short length."""
    for size in _PROBE_SIZES:
        rng = np.random.default_rng(size)
        run = 100.0 + np.cumsum(rng.normal(0.0, 1.0, size))
        gapped = run.copy()
        if size > 2:
            gapped[1] = np.nan
        flags = (np.arange(size) % 2 == 0).astype(float)
        for odd in _ODD:
            yield "ta_sum_kernel", (gapped, np.isfinite(gapped), odd)
            if odd >= 0:  # a negative window never returns in the Python loop (see the test below)
                yield "ta_var_kernel", (gapped, np.isfinite(gapped), odd)
            yield "ema_smooth_run", (run, odd, 0.5)
            yield "wilder_smooth_run", (run, odd)
            yield "wilder_rsi_smooth_run", (run, odd)
            yield "wilder_sum_run", (run, odd)
            yield "wilder_adx_smooth_run", (run, odd)
            for seed_offset in _ODD:
                yield "ema_fma_at_run", (run, odd, seed_offset)
            for start in (-1, 0, size, size + 1):
                yield "cci_kernel", (run, start, odd)
            yield "money_flow_index_kernel", (np.isfinite(gapped), np.abs(run), np.abs(run) * 0.5, odd)
            for start, first_output, other in itertools.product((-1, 0, size), (-1, 0, 2), (-1, 0, 3)):
                yield "chaikin_oscillator_kernel", (run, start, first_output, odd, other)
                yield "chaikin_oscillator_kernel", (run, start, first_output, other, odd)
        yield "cumulative_within_session_kernel", (gapped, flags)
        yield "dx_kernel", (np.abs(gapped), np.abs(run) * 0.5)
        highs, lows = run + 1.0, run - 1.0
        for first, stop in itertools.product((-3, -1, 0, size - 2, size), (-1, 0, size, size + 2)):
            yield "sar_run", (highs, lows, first, stop, np.full(size, np.nan))


def test_the_probe_table_covers_every_kernel() -> None:
    assert {name for name, _ in _probes()} == set(_REFERENCE)


@pytest.mark.parametrize("name", sorted(_REFERENCE))
def test_odd_periods_and_starts_meet_the_python_outcome(name: str) -> None:
    """Zero and negative periods: the Python loops wrap a negative index, divide by zero, or size a buffer
    negatively; the extension must answer the same bytes or raise the same exception type.
    """
    raised: set[type] = set()
    for case_name, arguments in _probes():
        if case_name != name:
            continue
        expected = _outcome(_python(name), arguments)
        assert _outcome(_COMPILED[name], arguments) == expected, (
            f"{name}{tuple(getattr(a, 'shape', a) for a in arguments)}"
        )
        if isinstance(expected, type):
            raised.add(expected)
    # The table must reach the error paths, or it proves nothing about them (three kernels never raise).
    if name not in {"cumulative_within_session_kernel", "dx_kernel", "ta_sum_kernel"}:
        assert raised, f"no probe made {name} raise"


def test_sar_run_writes_into_and_returns_the_array_it_was_given() -> None:
    highs, lows = np.linspace(10.0, 12.0, 6), np.linspace(9.0, 11.0, 6)
    out = np.full(6, np.nan)
    assert fnquant_kernels.sar_run(highs, lows, 0, 6, out) is out
    assert np.isfinite(out[1:]).all()


def test_a_strided_view_reads_as_its_contiguous_copy() -> None:
    series = np.linspace(1.0, 50.0, 40)[::2]
    assert not series.flags.c_contiguous
    assert _bits(fnquant_kernels.ema_smooth_run(series, 5, 1 / 3)) == _bits(
        _python("ema_smooth_run")(series.copy(), 5, 1 / 3)
    )


def test_a_panic_in_the_extension_is_an_exception_python_code_can_catch() -> None:
    """PyO3 raises a bare Rust panic as ``PanicException``, a ``BaseException`` that ``except Exception``
    lets through; the extension raises ``KernelPanic``, a ``RuntimeError``, instead.
    """
    assert issubclass(fnquant_kernels.KernelPanic, RuntimeError)
    with pytest.raises(fnquant_kernels.KernelPanic, match="deliberate panic"):
        _native._panic()
    with pytest.raises(Exception, match="deliberate panic") as caught:  # the claim: a plain Exception catches it
        _native._panic()
    assert caught.type is fnquant_kernels.KernelPanic


@pytest.mark.parametrize("window", [-1, -2, -3])
def test_ta_var_on_a_negative_window_returns_or_raises_where_the_python_loop_never_does(window: int) -> None:
    """The Python ``_ta_var_kernel`` walks its cursor backwards forever on a negative window, so there is no
    outcome to match; the extension must still end, with an ``Exception``.
    """
    series = np.linspace(1.0, 9.0, 9)
    with contextlib.suppress(Exception):  # ending in an Exception is the claim; a hang would time out
        fnquant_kernels.ta_var_kernel(series, np.isfinite(series), window)
