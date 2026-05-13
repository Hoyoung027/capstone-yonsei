#!/bin/bash
# BatchDecode KV tile_size_per_bdx 통합 실험
#
# 사용법:
#   bash run_decode_kv.sh
#   bash run_decode_kv.sh llama
#   bash run_decode_kv.sh llama3_8b qwen2.5_72b
#   TILE_SIZE_PER_BDX_VALS="1 2 3 4" bash run_decode_kv.sh llama3_8b
#   BATCH_SIZE=8 bash run_decode_kv.sh
#   nohup bash run_decode_kv.sh > results/logs/run_decode_kv_$(date +%Y%m%d).log 2>&1 &

set -e
cd "$(dirname "$0")"

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=(llama qwen gemma)
fi

if [ -n "${TILE_SIZE_PER_BDX_VALS:-}" ]; then
    read -r -a TILE_SIZE_PER_BDX_VALS <<< "$TILE_SIZE_PER_BDX_VALS"
else
    TILE_SIZE_PER_BDX_VALS=(1 2 3 4 5 6 7 8)
fi

BATCH_SIZE="${BATCH_SIZE:-8}"
PAGE_SIZE="${PAGE_SIZE:-16}"
PYTHON_BIN="${PYTHON_BIN:-/root/capstone-yonsei/venv/bin/python}"
KV_LENS="${KV_LENS:-$(seq -s ' ' 128 128 8192)}"
SKIP_CORRECTNESS="${SKIP_CORRECTNESS:-0}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

_restore_on_exit() { "$PYTHON_BIN" patch_decode.py restore 2>/dev/null || true; }
trap _restore_on_exit EXIT

clear_flashinfer_cache() {
    echo "  JIT cache 삭제..."
    rm -rf /root/.cache/flashinfer/
}

run_model() {
    local phase=$1 label=$2 qo=$3 kv=$4 dim=$5
    echo ""
    echo "  → phase=${phase}  label=${label}  heads=${qo}/${kv}  dim=${dim}  batch=${BATCH_SIZE}"
    local correctness_args=()
    if [ "$SKIP_CORRECTNESS" = "1" ]; then
        correctness_args=(--skip_correctness)
    fi
    "$PYTHON_BIN" -u test_decode_kv.py \
        --label "$label" \
        --num_qo_heads "$qo" --num_kv_heads "$kv" --head_dim "$dim" \
        --batch_size "$BATCH_SIZE" --page_size "$PAGE_SIZE" \
        --kv_lens "$KV_LENS" \
        "${correctness_args[@]}"
}

run_selected_models() {
    local phase=$1 label_mode=$2 tile=${3:-}

    unsupported_model() {
        local model=$1 group_size=$2
        echo "지원하지 않는 decode 모델 preset: ${model} (GROUP_SIZE=${group_size})"
        echo "현재 FlashInfer decode DISPATCH_GQA_GROUP_SIZE는 GROUP_SIZE 1, 2, 3, 4, 8만 지원합니다."
        exit 1
    }

    run_one() {
        local model=$1 qo=$2 kv=$3 dim=$4 label
        if [ "$label_mode" = "baseline" ]; then
            label="[${phase}] ${model}"
        else
            label="[experiment] ${model}_tile_size_per_bdx_${tile}"
        fi
        run_model "$phase" "$label" "$qo" "$kv" "$dim"
    }

    run_llama() {
        run_one "llama3_8b"   32  8 128
        run_one "llama3_70b"  64  8 128
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
    echo "######## ${phase}: FlashInfer auto ########"
    "$PYTHON_BIN" patch_decode.py restore
    clear_flashinfer_cache
    run_selected_models "$phase" baseline
}

run_forced_phase() {
    local tile=$1
    echo ""
    echo "######## forced tile_size_per_bdx=${tile} ########"
    "$PYTHON_BIN" patch_decode.py apply "$tile"
    clear_flashinfer_cache
    run_selected_models "forced_tile${tile}" experiment "$tile"
    "$PYTHON_BIN" patch_decode.py restore
}

mkdir -p results/logs results/data

echo "========================================"
echo " decode KV tile suite: ${TARGETS[*]}"
echo " order: baseline_before ${TILE_SIZE_PER_BDX_VALS[*]/#/tile} baseline_after"
echo " batch_size=${BATCH_SIZE} page_size=${PAGE_SIZE}"
echo " kv_lens=${KV_LENS}"
echo " python=${PYTHON_BIN}"
echo " skip_correctness=${SKIP_CORRECTNESS}"
echo " $(date)"
echo "========================================"

run_baseline_phase "baseline_before"

for tile in "${TILE_SIZE_PER_BDX_VALS[@]}"; do
    run_forced_phase "$tile"
done

run_baseline_phase "baseline_after"

echo ""
echo "========================================"
echo " 완료: $(date)"
echo "========================================"
