#!/bin/bash
# BatchDecode tensor-core NUM_MMA_KV tile experiment
#
# 사용법:
#   bash run_decode_tc.sh
#   bash run_decode_tc.sh llama
#   bash run_decode_tc.sh llama3_8b qwen2.5_72b
#   MMA_KV_VALS="1 2" bash run_decode_tc.sh llama3_8b
#   LABEL_SUFFIX="split_fixed_512" FIXED_SPLIT_SIZE=512 bash run_decode_tc.sh llama3_8b
#   LABEL_SUFFIX="split_k_4" TARGET_SPLIT_K=4 bash run_decode_tc.sh llama3_8b
#   DTYPE=bf16 bash run_decode_tc.sh llama3_8b
#   KV_LENS="128 1024" SKIP_CORRECTNESS=1 bash run_decode_tc.sh llama3_8b

set -e
cd "$(dirname "$0")"

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=(llama qwen gemma)
fi

if [ -n "${MMA_KV_VALS:-}" ]; then
    read -r -a MMA_KV_VALS <<< "$MMA_KV_VALS"
else
    MMA_KV_VALS=(1 2)
fi

BATCH_SIZE="${BATCH_SIZE:-8}"
PAGE_SIZE="${PAGE_SIZE:-16}"
BACKEND="${BACKEND:-fa2}"
DTYPE="${DTYPE:-float16}"
PYTHON_BIN="${PYTHON_BIN:-/root/capstone-yonsei/venv/bin/python}"
KV_LENS="${KV_LENS:-$(seq -s ' ' 128 128 8192)}"
SKIP_CORRECTNESS="${SKIP_CORRECTNESS:-0}"
FIXED_SPLIT_SIZE="${FIXED_SPLIT_SIZE:-}"
TARGET_SPLIT_K="${TARGET_SPLIT_K:-}"
DISABLE_SPLIT_KV="${DISABLE_SPLIT_KV:-0}"
LABEL_SUFFIX="${LABEL_SUFFIX:-}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

_restore_on_exit() { "$PYTHON_BIN" patch_decode_tc.py restore 2>/dev/null || true; }
trap _restore_on_exit EXIT

clear_flashinfer_cache() {
    echo "  전처리: JIT cache 삭제"
    rm -rf /root/.cache/flashinfer/
}

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

run_model() {
    local phase=$1 label=$2 qo=$3 kv=$4 dim=$5
    echo ""
    echo "  -> phase=${phase}  label=${label}  heads=${qo}/${kv}  dim=${dim}  batch=${BATCH_SIZE}"

    local args=()
    if [ -n "$FIXED_SPLIT_SIZE" ]; then
        args+=(--fixed_split_size "$FIXED_SPLIT_SIZE")
    fi
    if [ -n "$TARGET_SPLIT_K" ]; then
        args+=(--target_split_k "$TARGET_SPLIT_K")
    fi
    if [ "$DISABLE_SPLIT_KV" = "1" ]; then
        args+=(--disable_split_kv)
    fi
    if [ "$SKIP_CORRECTNESS" = "1" ]; then
        args+=(--skip_correctness)
    fi

    "$PYTHON_BIN" -u test_decode_tc.py \
        --label "$label" \
        --num_qo_heads "$qo" --num_kv_heads "$kv" --head_dim "$dim" \
        --batch_size "$BATCH_SIZE" --page_size "$PAGE_SIZE" \
        --kv_lens "$KV_LENS" --backend "$BACKEND" --dtype "$DTYPE" \
        "${args[@]}"
}

run_selected_models() {
    local phase=$1 label_mode=$2 mma=${3:-}

    unsupported_model() {
        local model=$1 group_size=$2
        echo "지원하지 않는 decode 모델 preset: ${model} (GROUP_SIZE=${group_size})"
        echo "현재 FlashInfer decode DISPATCH_GQA_GROUP_SIZE는 GROUP_SIZE 1, 2, 3, 4, 8만 지원합니다."
        exit 1
    }

    run_one() {
        local model=$1 qo=$2 kv=$3 dim=$4 label
        local model_label="$model"
        if [ -n "$LABEL_SUFFIX" ]; then
            model_label="${model}_${LABEL_SUFFIX}"
        fi
        if [ "$label_mode" = "baseline" ]; then
            label="[${phase}] ${model_label}"
        else
            label="[experiment] ${model_label}_num_mma_kv_${mma}"
        fi
        run_model "$phase" "$label" "$qo" "$kv" "$dim"
    }

    run_llama() {
        run_one "llama3_8b"   32 8 128
        run_one "llama3_70b"  64 8 128
    }
    run_qwen() {
        run_one "qwen2.5_72b" 64 8 128
    }
    run_gemma() {
        run_one "gemma2_9b"   16 8  256
        run_one "gemma2_27b"  32 16 128
    }

    for target in "${TARGETS[@]}"; do
        case "$target" in
            llama)        run_llama ;;
            qwen)         run_qwen ;;
            gemma)        run_gemma ;;
            llama3_8b)    run_one "llama3_8b"   32  8  128 ;;
            llama3_70b)   run_one "llama3_70b"  64  8  128 ;;
            llama3_405b)  unsupported_model "llama3_405b" 16 ;;
            qwen2.5_7b)   unsupported_model "qwen2.5_7b" 7 ;;
            qwen2.5_72b)  run_one "qwen2.5_72b" 64  8  128 ;;
            gemma2_9b)    run_one "gemma2_9b"   16  8  256 ;;
            gemma2_27b)   run_one "gemma2_27b"  32 16  128 ;;
            *)
                echo "알 수 없는 모델: $target"
                echo "사용 가능: llama | qwen | gemma | llama3_8b | llama3_70b | qwen2.5_72b | gemma2_9b | gemma2_27b"
                echo "decode 미지원 preset: llama3_405b(GROUP_SIZE=16), qwen2.5_7b(GROUP_SIZE=7)"
                exit 1
                ;;
        esac
    done
}

run_baseline_phase() {
    local phase=$1
    echo ""
    echo "######## ${phase}: FlashInfer tensor-core auto NUM_MMA_KV ########"
    "$PYTHON_BIN" patch_decode_tc.py restore
    clear_flashinfer_cache
    run_selected_models "$phase" baseline
}

run_forced_phase() {
    local mma=$1
    echo ""
    echo "######## forced tensor-core NUM_MMA_KV=${mma} ########"
    "$PYTHON_BIN" patch_decode_tc.py apply "$mma"
    clear_flashinfer_cache
    run_selected_models "forced_mma${mma}" experiment "$mma"
    "$PYTHON_BIN" patch_decode_tc.py restore
}

mkdir -p results/logs results/data results/plots

echo "========================================"
echo " decode tensor-core KV tile suite: ${TARGETS[*]}"
echo " order: baseline_before ${MMA_KV_VALS[*]/#/mma} baseline_after"
echo " batch_size=${BATCH_SIZE} page_size=${PAGE_SIZE} backend=${BACKEND}"
echo " dtype=${DTYPE}"
echo " fixed_split_size=${FIXED_SPLIT_SIZE:-none} target_split_k=${TARGET_SPLIT_K:-none} disable_split_kv=${DISABLE_SPLIT_KV}"
echo " label_suffix=${LABEL_SUFFIX:-none}"
echo " kv_lens=$(summarize_lens)"
echo " python=${PYTHON_BIN}"
echo " skip_correctness=${SKIP_CORRECTNESS}"
echo " $(date)"
echo "========================================"

run_baseline_phase "baseline_before"

for mma in "${MMA_KV_VALS[@]}"; do
    run_forced_phase "$mma"
done

run_baseline_phase "baseline_after"

echo ""
echo "========================================"
echo " 완료: $(date)"
echo "========================================"
