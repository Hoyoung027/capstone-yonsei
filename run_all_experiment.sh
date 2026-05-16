#!/usr/bin/env bash
# Run the main llama3 prefill/decode tile experiments in the background.
#
# Usage:
#   bash run_all_experiment.sh
#
# Logs are written under each experiment directory:
#   prefill_kv_tile_experiment/results/logs/
#   decode_kv_tile_experiment/results/logs/
#   decode_tensor_core_experiment/results/logs/
#
# Result CSV files are written by each experiment under:
#   prefill_kv_tile_experiment/results/data/
#   decode_kv_tile_experiment/results/data/
#   decode_tensor_core_experiment/results/data/
#
# The script starts one nohup job. Inside that job, every experiment runs
# sequentially to avoid GPU interference on a single-GPU machine.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/venv/bin/python}"
TARGETS=(llama3_8b llama3_70b)
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

run_step() {
    local name="$1"
    local dir="$2"
    shift 2

    local log_dir="${dir}/results/logs"
    local data_dir="${dir}/results/data"
    local log_path="${log_dir}/${name}_${RUN_ID}.log"

    mkdir -p "${log_dir}" "${data_dir}"

    echo ""
    echo "######## ${name} ########"
    echo "dir=${dir}"
    echo "log=${log_path}"
    echo "start=$(date)"

    cd "${dir}"
    "$@" > "${log_path}" 2>&1

    echo "done=$(date)"
}

run_decode_tc_split_k_correctness_sweep() {
    local kv_lens
    kv_lens="$(seq -s ' ' 128 128 8192)"

    echo "========================================"
    echo " decode tensor-core split-k correctness sweep"
    echo " model=llama3_8b"
    echo " batch_sizes=1 2 4 8 16"
    echo " split_modes=auto off fixed_128tok fixed_256tok fixed_512tok fixed_1024tok fixed_2048tok fixed_4096tok fixed_8192tok"
    echo " mma=auto 1 2"
    echo " page_size=16"
    echo " correctness=enabled"
    echo " kv_lens=128..8192 step 128"
    echo " start=$(date)"
    echo "========================================"

    "${PYTHON_BIN}" patch_decode_tc.py restore 2>/dev/null || true
    rm -rf /root/.cache/flashinfer/

    for bs in 1 2 4 8 16; do
        echo ""
        echo "======== batch_size=${bs}: split auto ========"
        env             BATCH_SIZE="${bs}"             LABEL_SUFFIX="fp16_split_auto_bs${bs}_correctness_full"             KV_LENS="${kv_lens}"             MMA_KV_VALS="1 2"             SKIP_CORRECTNESS=0             bash run_decode_tc.sh llama3_8b

        echo ""
        echo "======== batch_size=${bs}: split off ========"
        env             BATCH_SIZE="${bs}"             DISABLE_SPLIT_KV=1             LABEL_SUFFIX="fp16_split_off_bs${bs}_correctness_full"             KV_LENS="${kv_lens}"             MMA_KV_VALS="1 2"             SKIP_CORRECTNESS=0             bash run_decode_tc.sh llama3_8b

        for tok in 128 256 512 1024 2048 4096 8192; do
            local pages=$((tok / 16))
            echo ""
            echo "======== batch_size=${bs}: fixed split ${tok} tokens (${pages} pages) ========"
            env                 BATCH_SIZE="${bs}"                 FIXED_SPLIT_SIZE="${pages}"                 LABEL_SUFFIX="fp16_split_fixed_${tok}tok_bs${bs}_correctness_full"                 KV_LENS="${kv_lens}"                 MMA_KV_VALS="1 2"                 SKIP_CORRECTNESS=0                 bash run_decode_tc.sh llama3_8b
        done
    done

    "${PYTHON_BIN}" patch_decode_tc.py restore 2>/dev/null || true
    rm -rf /root/.cache/flashinfer/

    echo ""
    echo "completed=$(date)"
}

run_experiments() {
    cd "${ROOT_DIR}"
    export PYTHON_BIN
    export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

    echo "========================================"
    echo " run all experiments"
    echo " root=${ROOT_DIR}"
    echo " python=${PYTHON_BIN}"
    echo " targets=${TARGETS[*]}"
    echo " run_id=${RUN_ID}"
    echo " start=$(date)"
    echo " mode=sequential"
    echo "========================================"

    run_step         "prefill_kv_tile"         "${ROOT_DIR}/prefill_kv_tile_experiment"         bash run_tile_kv.sh "${TARGETS[@]}"

    run_step         "decode_kv_tile"         "${ROOT_DIR}/decode_kv_tile_experiment"         bash run_decode_kv.sh "${TARGETS[@]}"

    run_step         "decode_tensor_core_num_mma_kv"         "${ROOT_DIR}/decode_tensor_core_experiment"         bash run_decode_tc.sh "${TARGETS[@]}"

    run_step         "decode_tensor_core_split_k_correctness_full"         "${ROOT_DIR}/decode_tensor_core_experiment"         run_decode_tc_split_k_correctness_sweep

    echo ""
    echo "========================================"
    echo " all experiments completed: $(date)"
    echo "========================================"
    echo "prefill log: ${ROOT_DIR}/prefill_kv_tile_experiment/results/logs/prefill_kv_tile_${RUN_ID}.log"
    echo "decode log:  ${ROOT_DIR}/decode_kv_tile_experiment/results/logs/decode_kv_tile_${RUN_ID}.log"
    echo "tensor-core NUM_MMA_KV log: ${ROOT_DIR}/decode_tensor_core_experiment/results/logs/decode_tensor_core_num_mma_kv_${RUN_ID}.log"
    echo "tensor-core split-k correctness log: ${ROOT_DIR}/decode_tensor_core_experiment/results/logs/decode_tensor_core_split_k_correctness_full_${RUN_ID}.log"
}

if [[ "${1:-}" == "--child" ]]; then
    run_experiments
    exit 0
fi

mkdir -p \
    "${ROOT_DIR}/prefill_kv_tile_experiment/results/logs" \
    "${ROOT_DIR}/decode_kv_tile_experiment/results/logs" \
    "${ROOT_DIR}/decode_tensor_core_experiment/results/logs"

nohup bash "$0" --child > /dev/null 2>&1 &
PID=$!

echo "started run_all_experiment pid=${PID}"
echo "prefill log: ${ROOT_DIR}/prefill_kv_tile_experiment/results/logs/prefill_kv_tile_${RUN_ID}.log"
echo "decode log:  ${ROOT_DIR}/decode_kv_tile_experiment/results/logs/decode_kv_tile_${RUN_ID}.log"
echo "tensor-core NUM_MMA_KV log: ${ROOT_DIR}/decode_tensor_core_experiment/results/logs/decode_tensor_core_num_mma_kv_${RUN_ID}.log"
echo "tensor-core split-k correctness log: ${ROOT_DIR}/decode_tensor_core_experiment/results/logs/decode_tensor_core_split_k_correctness_full_${RUN_ID}.log"
echo "follow prefill: tail -f ${ROOT_DIR}/prefill_kv_tile_experiment/results/logs/prefill_kv_tile_${RUN_ID}.log"
echo "follow decode:  tail -f ${ROOT_DIR}/decode_kv_tile_experiment/results/logs/decode_kv_tile_${RUN_ID}.log"
echo "follow tensor-core: tail -f ${ROOT_DIR}/decode_tensor_core_experiment/results/logs/decode_tensor_core_num_mma_kv_${RUN_ID}.log"
echo "follow split-k: tail -f ${ROOT_DIR}/decode_tensor_core_experiment/results/logs/decode_tensor_core_split_k_correctness_full_${RUN_ID}.log"
