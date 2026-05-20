#!/usr/bin/env python3
"""Extract patched split-auto rows from the benchmark CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract patched heuristic rows from decode CSV.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label-contains", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.source)
    out = df[df["label"].astype(str).str.contains(args.label_contains, regex=False, na=False)].copy()
    if out.empty:
        raise SystemExit(f"no rows matched label substring: {args.label_contains}")
    out = out.sort_values(["batch_size", "kv_len", "label"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"saved {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
