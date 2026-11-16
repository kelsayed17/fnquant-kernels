# Contributing

## Setup

fnquant-kernels needs Python 3.14, a stable Rust toolchain (the minimum is 1.85) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run maturin develop --release
```

## Checks

Every change must pass `bash scripts/check.sh`, which CI runs on every pull request on Linux (x86-64 and arm64),
macOS and Windows:

```bash
cargo fmt --check
cargo clippy --locked --release -- -D warnings
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync ty check
uv run --no-sync pytest -q
```

CI also checks the crate builds on Rust 1.85 and audits the dependencies and the workflows.

- **Bit-exactness is the point.** Each kernel is a statement-for-statement translation of its plain-Python reference
  in `tests/reference.py`, with the same IEEE-754 operations in the same order: no fast-math, no implicit fused
  multiply-add. A change that moves any output by one ulp is a bug, even if it is "more accurate".
- **No unsafe code.** The crate is `#![forbid(unsafe_code)]`.
- **The TA-Lib oracle is pinned exactly.** Moving the `ta-lib` pin means re-deriving the references.

## Pull requests

Keep a pull request to one change, and describe what it changes for a caller of the kernels.
