#!/bin/bash
# baseline 실험 순차 실행
#
# 사용법:
#   bash run_baseline.sh llama              # LLaMA 3종만
#   bash run_baseline.sh qwen               # Qwen 2종만
#   bash run_baseline.sh gemma              # Gemma 2종만
#   bash run_baseline.sh llama qwen gemma   # 전체
#   bash run_baseline.sh                    # 전체 (인자 없으면 전체 실행)
#   nohup bash run_baseline.sh llama qwen gemma > run_baseline.log 2>&1 &. # 백그라운드 실행 예시


set -e
cd "$(dirname "$0")"

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=(llama qwen gemma)
fi

run_experiment() {
    local label=$1 qo=$2 kv=$3 dim=$4
    echo ""
    echo "  → label=${label}  heads=${qo}/${kv}  dim=${dim}"
    python test_tile_kv.py \
        --label "$label" \
        --num_qo_heads "$qo" --num_kv_heads "$kv" --head_dim "$dim" \
        --batch_size 8 --page_size 16
}

run_llama() {
    echo "===== LLaMA ====="
    run_experiment "llama3_8b"   32  8  128
    run_experiment "llama3_70b"  64  8  128
    run_experiment "llama3_405b" 128 8  128
}

run_qwen() {
    echo "===== Qwen ====="
    run_experiment "qwen2.5_7b"  28 4  128
    run_experiment "qwen2.5_72b" 64 8  128
}

run_gemma() {
    echo "===== Gemma ====="
    run_experiment "gemma2_9b"  16 8  256
    run_experiment "gemma2_27b" 32 16 128
}

echo "========================================"
echo " baseline experiments: ${TARGETS[*]}"
echo " $(date)"
echo "========================================"

for target in "${TARGETS[@]}"; do
    case "$target" in
        llama) run_llama ;;
        qwen)  run_qwen  ;;
        gemma) run_gemma ;;
        *)
            echo "알 수 없는 모델: $target  (llama | qwen | gemma)"
            exit 1
            ;;
    esac
done

echo ""
echo "========================================"
echo " 완료: $(date)"
echo "========================================"
