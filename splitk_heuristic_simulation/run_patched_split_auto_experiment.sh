#!/bin/bash
# Thin wrapper for the no-touch patched split-auto benchmark.
# Results are written only under splitk_heuristic_simulation/results/.

set -euo pipefail

ROOT="/root/capstone-yonsei"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/venv/bin/python}"

MODEL="${MODEL:-llama3_8b}"
BATCHES="${BATCHES:-1 2 4 8 16}"
KV_LENS="${KV_LENS:-$(seq -s ' ' 128 128 8192)}"
DTYPE="${DTYPE:-float16}"
ALPHA="${ALPHA:-16}"
BETA="${BETA:-16}"
CTA_TILE_KV_PROXY="${CTA_TILE_KV_PROXY:-64}"
WARMUP="${WARMUP:-100}"
REPEAT="${REPEAT:-100}"
TRIALS="${TRIALS:-10}"

"${PYTHON_BIN}" -u "${ROOT}/splitk_heuristic_simulation/run_patched_split_auto_experiment.py" \
  --model "${MODEL}" \
  --batches "${BATCHES}" \
  --kv-lens "${KV_LENS}" \
  --dtype "${DTYPE}" \
  --alpha "${ALPHA}" \
  --beta "${BETA}" \
  --cta-tile-kv-proxy "${CTA_TILE_KV_PROXY}" \
  --warmup "${WARMUP}" \
  --repeat "${REPEAT}" \
  --trials "${TRIALS}" \
  --skip-correctness
