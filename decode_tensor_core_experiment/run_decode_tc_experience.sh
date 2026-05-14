#!/bin/bash
# Decode tensor-core 종합 실험 스크립트.
#
# 수행 내용:
# - 여러 dtype(fp16, bf16)에 대해 실험을 순차 실행합니다.
# - 여러 모델(llama3_8b, llama3_70b, qwen2.5_72b, gemma2_9b, gemma2_27b)을 순차 실행합니다.
# - 각 모델/dtype마다 run_decode_tc_split_sweep.sh를 호출합니다.
# - split-k(auto/off/fixed_16~fixed_8192)와 NUM_MMA_KV(auto/1/2) 조합을 측정합니다.
# - 한 조합이 실패해도 기본적으로 다음 모델/dtype으로 넘어가고 마지막에 성공/실패를 요약합니다.
#
# Nohup:
#   mkdir -p results/logs
#   nohup bash run_decode_tc_experience.sh > results/logs/run_decode_tc_experience_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   tail -f "$(ls -t results/logs/run_decode_tc_experience*.log | head -n 1)"
#
# Defaults:
#   DTYPES="fp16 bf16"
#   MODELS="llama3_8b llama3_70b qwen2.5_72b gemma2_9b gemma2_27b"
#   SPLIT_MODES=study
#   KV_LENS="128 256 ... 8192"
#
# Custom:
#   DTYPES="fp16" MODELS="llama3_70b gemma2_27b" bash run_decode_tc_experience.sh

set -e
cd "$(dirname "$0")"

DTYPES="${DTYPES:-fp16 bf16}"
MODELS="${MODELS:-llama3_8b llama3_70b qwen2.5_72b gemma2_9b gemma2_27b}"
SPLIT_MODES="${SPLIT_MODES:-study}"
KV_LENS="${KV_LENS:-$(seq -s ' ' 128 128 8192)}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

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
echo " decode tensor-core full study queue"
echo " dtypes=${DTYPES}"
echo " models=${MODELS}"
echo " split_modes=${SPLIT_MODES}"
echo " kv_lens=$(summarize_lens)"
echo " stop_on_error=${STOP_ON_ERROR}"
echo " $(date)"
echo "========================================"

SUCCEEDED=()
FAILED=()

for dtype in $DTYPES; do
    for model in $MODELS; do
        ts="$(date +%Y%m%d_%H%M%S)"
        log="results/logs/run_decode_tc_split_study_${dtype}_${model}_${ts}.log"

        echo ""
        echo "######## start dtype=${dtype} model=${model} ########"
        echo " log=${log}"

        set +e
        env DTYPE="$dtype" SPLIT_MODES="$SPLIT_MODES" KV_LENS="$KV_LENS" \
            bash run_decode_tc_split_sweep.sh "$model" > "$log" 2>&1
        status=$?
        set -e

        if [ "$status" -eq 0 ]; then
            SUCCEEDED+=("${dtype}:${model}")
            echo "######## done dtype=${dtype} model=${model} ########"
        else
            FAILED+=("${dtype}:${model}:${status}")
            echo "######## failed dtype=${dtype} model=${model} exit=${status} ########"
            echo " last log lines:"
            tail -n 40 "$log" || true

            /root/capstone-yonsei/venv/bin/python patch_decode_tc.py restore 2>/dev/null || true
            rm -rf /root/.cache/flashinfer/ 2>/dev/null || true

            if [ "$STOP_ON_ERROR" = "1" ]; then
                echo "STOP_ON_ERROR=1 이므로 중단합니다."
                exit "$status"
            fi
        fi
    done
done

echo ""
echo "========================================"
echo " full study queue 완료: $(date)"
echo " succeeded: ${SUCCEEDED[*]:-none}"
echo " failed: ${FAILED[*]:-none}"
echo "========================================"
