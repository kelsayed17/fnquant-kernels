#!/usr/bin/env bash
# Every gate CI runs, in order. Expects `maturin develop` to have built the extension into the environment.
set -euo pipefail
cargo fmt --check
cargo clippy --locked --release -- -D warnings
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync ty check
uv run --no-sync pytest -q
