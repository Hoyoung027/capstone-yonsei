#!/usr/bin/env python3
"""Run real patched split-auto benchmarks into this folder only.

Unlike decode_tensor_core_experiment/run_decode_tc*.sh, this script never writes
to decode_tensor_core_experiment/results/data/decode_tc_results_fp16.csv.
It imports the benchmark runner, applies the FlashInfer scheduler patch, runs
split_auto only, and saves patched rows under splitk_heuristic_simulation/results.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import torch

from patch_flashinfer_splitk_heuristic import apply_patch, restore, status


ROOT = Path("/root/capstone-yonsei")
SIM_DIR = ROOT / "splitk_heuristic_simulation"
EXP_DIR = ROOT / "decode_tensor_core_experiment"
OUT_DIR = SIM_DIR / "results" / "data"

MODEL_PRESETS = {
    "llama3_8b": (32, 8, 128),
}


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.replace(",", " ").split() if x]


def load_test_decode_tc():
    sys.path.insert(0, str(EXP_DIR))
    spec = importlib.util.spec_from_file_location("decode_tc_bench", EXP_DIR / "test_decode_tc.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load test_decode_tc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dtype_suffix(dtype: str) -> str:
    if dtype in {"float16", "fp16"}:
        return "fp16"
    if dtype in {"bfloat16", "bf16"}:
        return "bf16"
    return dtype


def bench_ms_median(fn, warmup: int, repeat: int, trials: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    trial_ms = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        trial_ms.append(start.elapsed_time(end) / repeat)
    return float(torch.tensor(trial_ms).median().item())


def save_csv_replace_labels(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    existing_fields: list[str] = []
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
            existing = list(reader)

    labels = {r["label"] for r in rows}
    existing = [r for r in existing if r.get("label") not in labels]
    merged = existing + rows
    fields = list(dict.fromkeys(existing_fields + [k for r in rows for k in r.keys()]))

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run patched split-auto benchmark into simulation results.")
    parser.add_argument("--model", default="llama3_8b", choices=sorted(MODEL_PRESETS))
    parser.add_argument("--batches", default="1 2 4 8 16")
    parser.add_argument("--kv-lens", default=" ".join(str(x) for x in range(128, 8193, 128)))
    parser.add_argument("--dtype", default="float16", choices=["float16", "fp16", "bfloat16", "bf16"])
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--backend", default="fa2")
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--beta", type=int, default=16)
    parser.add_argument("--cta-tile-kv-proxy", type=int, default=64)
    parser.add_argument("--skip-correctness", action="store_true", default=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    qo_heads, kv_heads, head_dim = MODEL_PRESETS[args.model]
    batches = parse_int_list(args.batches)
    kv_lens = parse_int_list(args.kv_lens)
    out_csv = args.output or (
        OUT_DIR / f"decode_tc_results_{dtype_suffix(args.dtype)}_patched_split_auto_alpha{args.alpha}_beta{args.beta}.csv"
    )

    bench = load_test_decode_tc()
    dtype = bench.parse_dtype(args.dtype)

    def patched_bench_ms(fn):
        return bench_ms_median(fn, warmup=args.warmup, repeat=args.repeat, trials=args.trials)

    bench.bench_ms = patched_bench_ms

    print("========================================")
    print(" patched split-auto benchmark")
    print(f" model={args.model}")
    print(f" batches={batches}")
    print(f" kv_lens={kv_lens[0]}..{kv_lens[-1]} ({len(kv_lens)} values)")
    print(f" alpha={args.alpha} beta={args.beta} cta_tile_kv_proxy={args.cta_tile_kv_proxy}")
    print(f" output={out_csv}")
    print(f" measurement=median over trials={args.trials}, repeat={args.repeat}, warmup={args.warmup}")
    print(" This script does not write to decode_tensor_core_experiment/results/data.")
    print("========================================")

    all_rows: list[dict] = []
    try:
        apply_patch(args.alpha, args.beta, args.cta_tile_kv_proxy)
        shutil.rmtree("/root/.cache/flashinfer", ignore_errors=True)

        for batch in batches:
            label = (
                f"[patched_heuristic] {args.model}_{args.dtype}_"
                f"split_auto_alpha{args.alpha}_beta{args.beta}_bs{batch}"
            )
            print("")
            print(f"---- {label} ----")
            rows = bench.run(
                label=label,
                num_qo_heads=qo_heads,
                num_kv_heads=kv_heads,
                head_dim=head_dim,
                batch_size=batch,
                page_size=args.page_size,
                kv_lens=kv_lens,
                backend=args.backend,
                dtype=dtype,
                fixed_split_size=None,
                disable_split_kv=False,
                target_split_k=None,
                skip_correctness=args.skip_correctness,
            )
            for row in rows:
                row["measurement_mode"] = "median"
                row["measurement_warmup"] = args.warmup
                row["measurement_repeat"] = args.repeat
                row["measurement_trials"] = args.trials
            all_rows.extend(rows)
    finally:
        restore()
        status()

    save_csv_replace_labels(all_rows, out_csv)
    print("")
    print(f"saved patched rows: {len(all_rows)}")
    print(f"saved patched CSV: {out_csv}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
