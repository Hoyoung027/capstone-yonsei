#!/bin/bash
# Sweep split-k settings for tensor-core decode NUM_MMA_KV experiments.
# 
# 실험 동작
# nohup env SPLIT_MODES=study KV_LENS="$(seq -s ' ' 128 128 8192)" bash run_decode_tc_split_sweep.sh llama3_8b > results/logs/run_decode_tc_split_study_llama3_8b_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 실험 조합
# split_auto    + MMA auto, 1, 2
# split_off     + MMA auto, 1, 2
# fixed_16      + MMA auto, 1, 2
# fixed_32      + MMA auto, 1, 2
# fixed_64      + MMA auto, 1, 2
# fixed_128     + MMA auto, 1, 2
# fixed_256     + MMA auto, 1, 2
# fixed_512     + MMA auto, 1, 2
# fixed_1024    + MMA auto, 1, 2
# fixed_2048    + MMA auto, 1, 2
# fixed_4096    + MMA auto, 1, 2
# fixed_8192    + MMA auto, 1, 2

#
# Usage:
#   bash run_decode_tc_split_sweep.sh llama3_8b
#   SPLIT_MODES="auto off fixed_256 fixed_512 fixed_1024 fixed_2048" bash run_decode_tc_split_sweep.sh llama3_8b
#   SPLIT_MODES=full bash run_decode_tc_split_sweep.sh llama3_8b
#   SPLIT_MODES=all KV_LENS="$(seq -s ' ' 128 128 8192)" bash run_decode_tc_split_sweep.sh llama3_8b
#   SPLIT_MODES=study KV_LENS="$(seq -s ' ' 128 128 8192)" bash run_decode_tc_split_sweep.sh llama3_8b
#   STOP_ON_ERROR=1 bash run_decode_tc_split_sweep.sh llama3_8b
#
# Nohup:
#   mkdir -p results/logs
#   nohup bash run_decode_tc_split_sweep.sh llama3_8b > results/logs/run_decode_tc_split_sweep_llama3_8b_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   nohup env SPLIT_MODES=all KV_LENS="$(seq -s ' ' 128 128 8192)" bash run_decode_tc_split_sweep.sh llama3_8b > results/logs/run_decode_tc_split_all_llama3_8b_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   nohup env SPLIT_MODES=study KV_LENS="$(seq -s ' ' 128 128 8192)" bash run_decode_tc_split_sweep.sh llama3_8b > results/logs/run_decode_tc_split_study_llama3_8b_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   tail -f "$(ls -t results/logs/run_decode_tc_split*.log | head -n 1)"
#

set -e
cd "$(dirname "$0")"

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=(llama3_8b)
fi

case "${SPLIT_MODES:-pilot}" in
    pilot)
        SPLIT_MODES="auto off fixed_256 fixed_512 fixed_1024 fixed_2048"
        ;;
    full)
        SPLIT_MODES="auto off fixed_16 fixed_32 fixed_64 fixed_128 fixed_256 fixed_512 fixed_1024 fixed_2048"
        ;;
    all|exhaustive|study|proper)
        SPLIT_MODES="auto off fixed_16 fixed_32 fixed_64 fixed_128 fixed_256 fixed_512 fixed_1024 fixed_2048 fixed_4096 fixed_8192"
        ;;
    *)
        # Custom list, for example:
        #   SPLIT_MODES="auto off fixed_512 fixed_1024"
        ;;
esac

MMA_KV_VALS="${MMA_KV_VALS:-1 2}"
PYTHON_BIN="${PYTHON_BIN:-/root/capstone-yonsei/venv/bin/python}"
KV_LENS="${KV_LENS:-128 512 1024 2048 4096 8192}"
SKIP_CORRECTNESS="${SKIP_CORRECTNESS:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

export MMA_KV_VALS PYTHON_BIN KV_LENS SKIP_CORRECTNESS

mkdir -p results/logs results/data results/plots

echo "========================================"
echo " decode tensor-core split-k sweep"
echo " targets=${TARGETS[*]}"
echo " split_modes=${SPLIT_MODES}"
echo " mma_kv_vals=${MMA_KV_VALS}"
echo " kv_lens=${KV_LENS}"
echo " skip_correctness=${SKIP_CORRECTNESS}"
echo " stop_on_error=${STOP_ON_ERROR}"
echo " python=${PYTHON_BIN}"
echo " $(date)"
echo "========================================"

SUCCEEDED_MODES=()
FAILED_MODES=()

for mode in $SPLIT_MODES; do
    echo ""
    echo "######## split mode: ${mode} ########"

    unset FIXED_SPLIT_SIZE
    export DISABLE_SPLIT_KV=0

    case "$mode" in
        auto)
            export LABEL_SUFFIX="split_auto"
            ;;
        off|disable|disabled)
            export LABEL_SUFFIX="split_off"
            export DISABLE_SPLIT_KV=1
            ;;
        fixed_*)
            size="${mode#fixed_}"
            if ! [[ "$size" =~ ^[0-9]+$ ]]; then
                echo "잘못된 split mode: ${mode}"
                exit 1
            fi
            export LABEL_SUFFIX="split_fixed_${size}"
            export FIXED_SPLIT_SIZE="$size"
            ;;
        *)
            echo "알 수 없는 split mode: ${mode}"
            echo "사용 가능: auto off fixed_N 또는 SPLIT_MODES=pilot|full|all|exhaustive|study|proper"
            exit 1
            ;;
    esac

    set +e
    bash run_decode_tc.sh "${TARGETS[@]}"
    status=$?
    set -e

    if [ "$status" -eq 0 ]; then
        SUCCEEDED_MODES+=("$mode")
        echo "######## split mode 완료: ${mode} ########"
    else
        FAILED_MODES+=("${mode}:${status}")
        echo "######## split mode 실패: ${mode} (exit=${status}) ########"
        "$PYTHON_BIN" patch_decode_tc.py restore 2>/dev/null || true
        rm -rf /root/.cache/flashinfer/ 2>/dev/null || true

        if [ "$STOP_ON_ERROR" = "1" ]; then
            echo "STOP_ON_ERROR=1 이므로 sweep을 중단합니다."
            exit "$status"
        fi
    fi
done

unset LABEL_SUFFIX FIXED_SPLIT_SIZE DISABLE_SPLIT_KV

echo ""
echo "========================================"
echo " split-k sweep 완료: $(date)"
echo " succeeded: ${SUCCEEDED_MODES[*]:-none}"
echo " failed: ${FAILED_MODES[*]:-none}"
echo "========================================"
