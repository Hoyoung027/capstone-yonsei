#!/bin/bash
# Quick smoke test for one tensor-core decode tile experiment target.
#
# Usage:
#   bash smoke_decode_tc.sh
#   bash smoke_decode_tc.sh llama3_70b
#   KV_LENS="128 1024 2048" MMA_KV_VALS="1 2" bash smoke_decode_tc.sh llama3_8b

set -e
cd "$(dirname "$0")"

TARGET="${1:-llama3_8b}"

export KV_LENS="${KV_LENS:-128 1024}"
export MMA_KV_VALS="${MMA_KV_VALS:-1 2}"
export SKIP_CORRECTNESS="${SKIP_CORRECTNESS:-1}"

echo "========================================"
echo " smoke decode tensor-core KV tile"
echo " target=${TARGET}"
echo " kv_lens=${KV_LENS}"
echo " mma_kv_vals=${MMA_KV_VALS}"
echo " skip_correctness=${SKIP_CORRECTNESS}"
echo "========================================"

bash run_decode_tc.sh "$TARGET"
