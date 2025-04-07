#!/bin/sh
set -e
python -m black --check bench_runner tests
python -m flake8 bench_runner tests
python -m pyright -p pyproject.toml bench_runner
