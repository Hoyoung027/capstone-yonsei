#!/usr/bin/env python3
"""Copy FlashInfer default and oracle-reference rows into this folder.

This keeps the splitk_heuristic_simulation folder self-contained:
  - default split_auto rows
  - k_1..k_20 rows used to compute oracle

No source CSV is modified.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path("/root/capstone-yonsei")
SOURCE = ROOT / "decode_tensor_core_experiment" / "results" / "data" / "decode_tc_results_fp16.csv"
OUT = ROOT / "splitk_heuristic_simulation" / "results" / "data" / "decode_tc_results_fp16_reference_default_oracle.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reference rows for patched split-k comparison.")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--batches", default="1 2 4 8 16")
    args = parser.parse_args()

    batches = {int(x) for x in args.batches.replace(",", " ").split()}
    df = pd.read_csv(args.source)
    pattern = re.compile(
        rf"\[(baseline_before|baseline_after)\]\s+{re.escape(args.model)}_float16_split_(auto|k_\d+)_bs(\d+)$"
    )

    keep = []
    for label in df["label"].astype(str):
        m = pattern.fullmatch(label)
        keep.append(bool(m and int(m.group(3)) in batches))

    out = df[keep].copy()
    if out.empty:
        raise SystemExit("no reference rows matched")
    out = out.sort_values(["batch_size", "label", "kv_len"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"saved {len(out)} reference rows: {args.output}")
    print(out.groupby("batch_size")["kv_len"].nunique().to_string())


if __name__ == "__main__":
    main()
