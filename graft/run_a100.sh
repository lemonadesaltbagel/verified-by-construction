#!/usr/bin/env bash
# HISTORICAL: ran with the superseded circuit, now `--circuit pattern_v1` (PHASE3_DESIGN.md §10-13).
# Phase 3.4 on an A100 (80 GB recommended): OPT-6.7B, float32 main run + float64 control.
# Requirements: >= 64 GB CPU RAM (the HF model is loaded on CPU in float32, ~27 GB), ~30 GB disk for weights.
# Setup:  cd graft && uv sync   (CUDA torch from pyproject/uv.lock)
cd "$(dirname "$0")"
mkdir -p results
run() { echo "=== $* $(date '+%F %T')"; uv run python opt_graft.py "$@" 2>&1 | grep -v -i "warn\|it/s\]" > "results/$(basename $1)_$3.log"; tail -2 "results/$(basename $1)_$3.log"; nvidia-smi --query-gpu=memory.used --format=csv,noheader; }
run facebook/opt-6.7b --dtype float32 --device cuda --windows 80 --win 512 --ntrig 80
run facebook/opt-6.7b --dtype float64 --device cuda --windows 10 --win 512 --ntrig 10   # needs ~60 GB GPU
echo ALL_DONE
