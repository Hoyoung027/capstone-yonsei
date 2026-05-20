# Split-k Heuristic Simulation

This folder evaluates a proposed FlashInfer split-k scheduler rule without
patching or rebuilding FlashInfer.

The idea is to use the existing sweep CSV:

```text
decode_tensor_core_experiment/results/data/decode_tc_results_fp16.csv
```

The CSV already contains measured latency for:

```text
FlashInfer split_auto
k_1, k_2, ..., k_20
```

So we can simulate a new scheduler by selecting one of the already measured
`k_N` rows for every `(batch_size, kv_len)` point.

## Compared Policies

```text
FlashInfer default
  split_auto, NUM_MMA_KV auto

Proposed heuristic
  choose k_N using a minimum-work-per-chunk cap

Oracle
  choose the fastest measured k_1..k_20 at each kv_len
```

This isolates split-k scheduling. `NUM_MMA_KV` is kept on auto for all three
policies.

## Proposed Rule

FlashInfer auto tries to create enough CTA work using split-k. The proposed
rule keeps that occupancy-oriented choice but prevents over-splitting short KV
sequences:

```text
work_cap = floor(kv_len / (alpha * CTA_TILE_KV))
proposed_chunks = min(FlashInfer_default_chunks, work_cap)
```

Interpretation:

```text
Each KV chunk should contain at least alpha x CTA_TILE_KV tokens.
```

For example, when `CTA_TILE_KV=64`:

```text
alpha=4 -> each chunk should have at least 256 tokens
alpha=8 -> each chunk should have at least 512 tokens
```

The script maps `proposed_chunks` to the closest available `k_N` measurement.

## Run

```bash
cd /root/capstone-yonsei

/root/capstone-yonsei/venv/bin/python \
  splitk_heuristic_simulation/simulate_splitk_heuristic.py \
  --model llama3_8b \
  --baseline before \
  --alphas "2 4 8 16"
```

Optional batch-aware cap:

```bash
/root/capstone-yonsei/venv/bin/python \
  splitk_heuristic_simulation/simulate_splitk_heuristic.py \
  --model llama3_8b \
  --baseline before \
  --alphas "2 4 8 16" \
  --beta 16
```

## Outputs

Results are written to:

```text
splitk_heuristic_simulation/results/
```

Main files:

```text
llama3_8b_heuristic_alpha_sweep.csv
llama3_8b_alpha_<alpha>_proposed_details.csv
llama3_8b_alpha_<alpha>_proposed_vs_default_oracle_geomean.png
llama3_8b_bs<batch>_alpha_<alpha>_proposed_detail.png
```

The geomean plot is the main presentation figure. It shows how much of the
oracle split-k gain is recovered by the proposed heuristic.

## Real Patched FlashInfer Experiment

After the CSV simulation, run the actual patched FlashInfer split-auto path.
This experiment stores patched results inside this folder and restores the
original `decode_tensor_core_experiment` CSV afterward.

```bash
cd /root/capstone-yonsei

nohup bash splitk_heuristic_simulation/run_patched_split_auto_experiment.sh \
  > splitk_heuristic_simulation/results/logs/run_patched_split_auto_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Default settings:

```text
model = llama3_8b
batch sizes = 1, 2, 4, 8, 16
kv_len = 128..8192, step 128
alpha = 16
beta = 16
CTA_TILE_KV proxy = 64
dtype = float16
backend = fa2
correctness = skipped
```

The patched CSV is written to:

```text
splitk_heuristic_simulation/results/data/decode_tc_results_fp16_patched_split_auto_alpha16_beta16.csv
```

This runner does not write to `decode_tensor_core_experiment/results/data`. It imports the benchmark `run()` function, applies the FlashInfer scheduler patch, runs patched split_auto for each batch, saves rows directly under this folder, and restores the scheduler.


Prepare a self-contained reference CSV from the existing default/oracle sweep:

```bash
/root/capstone-yonsei/venv/bin/python \
  splitk_heuristic_simulation/prepare_reference_results.py \
  --model llama3_8b
```

This copies only FlashInfer `split_auto` and `k_1..k_20` baseline rows into:

```text
splitk_heuristic_simulation/results/data/decode_tc_results_fp16_reference_default_oracle.csv
```

The original CSV is not modified.

After the patched benchmark finishes, compare real patched results against the
stored FlashInfer default and oracle:

```bash
/root/capstone-yonsei/venv/bin/python \
  splitk_heuristic_simulation/compare_patched_results.py \
  --model llama3_8b
```

Main real-benchmark comparison outputs:

```text
splitk_heuristic_simulation/results/data/llama3_8b_real_patched_comparison_summary.csv
splitk_heuristic_simulation/results/data/llama3_8b_real_patched_comparison_details.csv
splitk_heuristic_simulation/results/plots/llama3_8b_real_patched_vs_default_oracle_geomean.png
splitk_heuristic_simulation/results/plots/llama3_8b_bs<batch>_real_patched_detail.png
```

Patch status and manual restore:

```bash
/root/capstone-yonsei/venv/bin/python splitk_heuristic_simulation/patch_flashinfer_splitk_heuristic.py status
/root/capstone-yonsei/venv/bin/python splitk_heuristic_simulation/patch_flashinfer_splitk_heuristic.py restore
```

