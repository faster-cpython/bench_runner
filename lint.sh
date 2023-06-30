#!/bin/sh
set -e
python -m black --check bench_runner tests --exclude summarize_stats\\.py
python -m flake8 bench_runner tests --exclude summarize_stats.py
python -m pyright bench_runner
