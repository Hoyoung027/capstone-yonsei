#!/bin/bash
# Tensor-core decode split-k count sweep.
#
# 이 스크립트는 /root/capstone-yonsei/decode_tensor_core_experiment 안에서 실행하는 것을 기준으로 한다.
#
# 주요 실험 대상:
#   - 모델: llama3_8b
#   - split-k auto + split-k count k: 1..20, SPLIT_MODES=k_study 또는 "auto k_1 ... k_20" 형태로 지정
#   - batch size: 1, 2, 4, 8, 16
#   - NUM_MMA_KV: auto baseline + 1 강제 + 2 강제
#   - kv_len: 128..8192, 128 간격
#   - correctness: SKIP_CORRECTNESS=0이면 수행
#
# 내부 동작:
#   k_N은 각 kv_len마다 FlashInfer fixed_split_size(page 단위)로 변환된다.
#   fixed_split_size_pages = ceil(kv_len / (N * page_size))
#   CSV에는 요청한 split_k_count와 FlashInfer plan에서 읽은 실제 num_chunks_kv가 함께 저장된다.
#
# 단일 batch 실행:
#   env SPLIT_MODES=k_study KV_LENS="$(seq -s ' ' 128 128 8192)" BATCH_SIZE=1 MMA_KV_VALS="1 2" SKIP_CORRECTNESS=0 bash run_decode_tc_split_sweep.sh llama3_8b
#
# 모든 batch를 nohup으로 순차 실행:
#   mkdir -p results/logs
#   nohup bash -lc 'for bs in 1 2 4 8 16; do env SPLIT_MODES=k_study KV_LENS="$(seq -s '\'' '\'' 128 128 8192)" BATCH_SIZE="$bs" MMA_KV_VALS="1 2" SKIP_CORRECTNESS=0 bash run_decode_tc_split_sweep.sh llama3_8b; done' > results/logs/run_decode_tc_split_auto_k1_20_all_batches_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# 최신 로그 확인:
#   tail -f "$(ls -t results/logs/run_decode_tc_split_auto_k1_20_all_batches_*.log | head -n 1)"

set -e
cd "$(dirname "$0")"

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
    if [ -n "${MODEL:-}" ]; then
        read -r -a TARGETS <<< "$MODEL"
    else
        TARGETS=(llama3_8b)
    fi
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
    k_study|splitk_study|count_study)
        SPLIT_MODES="auto $(seq -f 'k_%g' -s ' ' 1 20)"
        ;;
    *)
        # Custom list, for example:
        #   SPLIT_MODES="auto off fixed_512 fixed_1024"
        ;;
esac

MMA_KV_VALS="${MMA_KV_VALS:-1 2}"
DTYPE="${DTYPE:-float16}"
PYTHON_BIN="${PYTHON_BIN:-/root/capstone-yonsei/venv/bin/python}"
KV_LENS="${KV_LENS:-128 512 1024 2048 4096 8192}"
SKIP_CORRECTNESS="${SKIP_CORRECTNESS:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

export MMA_KV_VALS DTYPE PYTHON_BIN KV_LENS SKIP_CORRECTNESS

mkdir -p results/logs results/data results/plots

summarize_lens() {
    local arr=($KV_LENS)
    local n=${#arr[@]}
    if [ "$n" -eq 0 ]; then
        echo "empty"
    elif [ "$n" -eq 1 ]; then
        echo "${arr[0]}"
    else
        local step=$((arr[1] - arr[0]))
        echo "${arr[0]}..${arr[$((n - 1))]} step ${step} (${n} values)"
    fi
}

echo "========================================"
echo " decode tensor-core split-k sweep"
echo " targets=${TARGETS[*]}"
echo " split_modes=${SPLIT_MODES}"
echo " mma_kv_vals=${MMA_KV_VALS}"
echo " dtype=${DTYPE}"
echo " kv_lens=$(summarize_lens)"
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

    unset FIXED_SPLIT_SIZE TARGET_SPLIT_K
    export DISABLE_SPLIT_KV=0

    case "$mode" in
        auto)
            export LABEL_SUFFIX="${DTYPE}_split_auto_bs${BATCH_SIZE:-8}"
            ;;
        off|disable|disabled)
            export LABEL_SUFFIX="${DTYPE}_split_off_bs${BATCH_SIZE:-8}"
            export DISABLE_SPLIT_KV=1
            ;;
        fixed_*)
            size="${mode#fixed_}"
            if ! [[ "$size" =~ ^[0-9]+$ ]]; then
                echo "잘못된 split mode: ${mode}"
                exit 1
            fi
            export LABEL_SUFFIX="${DTYPE}_split_fixed_${size}_bs${BATCH_SIZE:-8}"
            export FIXED_SPLIT_SIZE="$size"
            ;;
        k_*|splitk_*|count_*)
            count="${mode#k_}"
            count="${count#splitk_}"
            count="${count#count_}"
            if ! [[ "$count" =~ ^[0-9]+$ ]] || [ "$count" -le 0 ]; then
                echo "잘못된 split-k count mode: ${mode}"
                exit 1
            fi
            export LABEL_SUFFIX="${DTYPE}_split_k_${count}_bs${BATCH_SIZE:-8}"
            export TARGET_SPLIT_K="$count"
            ;;
        *)
            echo "알 수 없는 split mode: ${mode}"
            echo "사용 가능: auto off fixed_N k_N splitk_N count_N 또는 SPLIT_MODES=pilot|full|all|exhaustive|study|proper"
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

unset LABEL_SUFFIX FIXED_SPLIT_SIZE TARGET_SPLIT_K DISABLE_SPLIT_KV

echo ""
echo "========================================"
echo " split-k sweep 완료: $(date)"
echo " succeeded: ${SUCCEEDED_MODES[*]:-none}"
echo " failed: ${FAILED_MODES[*]:-none}"
echo "========================================"
