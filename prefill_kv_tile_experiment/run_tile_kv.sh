#!/bin/bash
# Baseline/forced NUM_MMA_KV 통합 실험
#
# 순서:
#   1) FlashInfer auto baseline_before
#   2) forced NUM_MMA_KV values
#   3) FlashInfer auto baseline_after
#
# 사용법:
#   bash run_tile_kv.sh
#   bash run_tile_kv.sh llama
#   bash run_tile_kv.sh llama3_8b qwen2.5_7b
#   MMA_KV_VALS="1 2 3 4 5 6 7" bash run_tile_kv.sh llama3_8b
#   nohup bash run_tile_kv.sh > results/logs/run_tile_kv_$(date +%Y%m%d).log 2>&1 &

set -e
cd "$(dirname "$0")"

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=(llama qwen gemma)
fi

if [ -n "${MMA_KV_VALS:-}" ]; then
    read -r -a MMA_KV_VALS <<< "$MMA_KV_VALS"
else
    MMA_KV_VALS=(1 2 4 8)
fi

# 예외 상황에서도 가상환경 안의 prefill.cuh를 원본 상태로 되돌린다.
_restore_on_exit() { python patch_prefill.py restore 2>/dev/null || true; }
trap _restore_on_exit EXIT

clear_flashinfer_cache() {
    echo "  JIT cache 삭제..."
    rm -rf /root/.cache/flashinfer/
}

run_model() {
    local phase=$1 label=$2 qo=$3 kv=$4 dim=$5
    echo ""
    echo "  → phase=${phase}  label=${label}  heads=${qo}/${kv}  dim=${dim}"
    python -u test_tile_kv.py \
        --label "$label" \
        --num_qo_heads "$qo" --num_kv_heads "$kv" --head_dim "$dim" \
        --batch_size 8 --page_size 16
}

run_selected_models() {
    local phase=$1 label_mode=$2 mma=${3:-}

    run_one() {
        local model=$1 qo=$2 kv=$3 dim=$4 label
        if [ "$label_mode" = "baseline" ]; then
            label="[${phase}] ${model}"
        else
            label="[experiment] ${model}_num_mma_kv_${mma}"
        fi
        run_model "$phase" "$label" "$qo" "$kv" "$dim"
    }

    run_llama() {
        run_one "llama3_8b"   32  8  128
        run_one "llama3_70b"  64  8  128
        run_one "llama3_405b" 128 8  128
    }
    run_qwen() {
        run_one "qwen2.5_7b"  28 4 128
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
            llama3_405b)  run_one "llama3_405b" 128 8  128 ;;
            qwen2.5_7b)   run_one "qwen2.5_7b"  28  4  128 ;;
            qwen2.5_72b)  run_one "qwen2.5_72b" 64  8  128 ;;
            gemma2_9b)    run_one "gemma2_9b"   16  8  256 ;;
            gemma2_27b)   run_one "gemma2_27b"  32 16  128 ;;
            *)
                echo "알 수 없는 모델: $target"
                echo "사용 가능: llama | qwen | gemma | llama3_8b | llama3_70b | llama3_405b | qwen2.5_7b | qwen2.5_72b | gemma2_9b | gemma2_27b"
                exit 1
                ;;
        esac
    done
}

run_baseline_phase() {
    local phase=$1
    echo ""
    echo "######## ${phase}: FlashInfer auto ########"
    python patch_prefill.py restore
    clear_flashinfer_cache
    run_selected_models "$phase" baseline
}

run_forced_phase() {
    local mma=$1
    echo ""
    echo "######## forced NUM_MMA_KV=${mma} ########"
    python patch_prefill.py apply "$mma"
    clear_flashinfer_cache
    run_selected_models "forced_mma${mma}" experiment "$mma"
    python patch_prefill.py restore
}

echo "========================================"
echo " tile KV suite: ${TARGETS[*]}"
echo " order: baseline_before ${MMA_KV_VALS[*]/#/mma} baseline_after"
echo " correctness: enabled for every run"
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
