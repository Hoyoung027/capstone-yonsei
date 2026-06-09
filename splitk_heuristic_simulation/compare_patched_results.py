#!/usr/bin/env python3
"""Compare real patched split-auto results against default and oracle CSV data."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = (
    ROOT
    / "splitk_heuristic_simulation"
    / "results"
    / "data"
    / "decode_tc_results_fp16_reference_default_oracle.csv"
)
PATCHED_CSV = (
    ROOT
    / "splitk_heuristic_simulation"
    / "results"
    / "data"
    / "decode_tc_results_fp16_patched_split_auto_alpha16_beta16.csv"
)
OUT_DIR = ROOT / "splitk_heuristic_simulation" / "results" / "plots"

DEFAULT_COLOR = "#1f77b4"
PATCHED_COLOR = "#2ca02c"
ORACLE_COLOR = "#ff7f0e"

GEOMEAN_DEFAULT_COLOR = "#8c8c8c"
GEOMEAN_PATCHED_COLOR = "#4c78a8"
GEOMEAN_ORACLE_COLOR = "#f58518"


def geomean(values: pd.Series) -> float:
    values = values.dropna()
    values = values[values > 0]
    return float(np.exp(np.log(values).mean()))


def parse_default_label(label: str):
    m = re.fullmatch(r"\[(baseline_before|baseline_after)\]\s+(.+?)_float16_split_(auto|k_\d+)_bs(\d+)", label)
    if not m:
        return None
    phase, model, split, batch = m.groups()
    return phase, model, split, int(batch)


def prepare_default(df: pd.DataFrame, model: str) -> pd.DataFrame:
    parsed = df["label"].map(parse_default_label)
    keep = parsed.notna()
    out = df[keep].copy()
    parsed = parsed[keep]
    out["phase"] = parsed.map(lambda x: x[0])
    out["base_model"] = parsed.map(lambda x: x[1])
    out["split_mode"] = parsed.map(lambda x: x[2])
    out["condition_batch_size"] = parsed.map(lambda x: x[3])
    return out[out["base_model"] == model].copy()


def build_comparison(default_df: pd.DataFrame, patched_df: pd.DataFrame, model: str, batches: list[int]) -> pd.DataFrame:
    rows = []
    for bs in batches:
        base = default_df[
            (default_df["condition_batch_size"] == bs)
            & (default_df["split_mode"] == "auto")
            & (default_df["phase"] == "baseline_before")
        ].set_index("kv_len")
        patched = patched_df[patched_df["batch_size"] == bs].set_index("kv_len")
        if base.empty or patched.empty:
            raise SystemExit(f"missing base or patched rows for BS={bs}")

        k_rows = default_df[
            (default_df["condition_batch_size"] == bs)
            & (default_df["split_mode"].str.match(r"k_\d+", na=False))
            & (default_df["phase"] == "baseline_before")
        ].copy()
        if k_rows.empty:
            raise SystemExit(f"missing oracle k_N rows for BS={bs}")
        k_rows["split_k"] = k_rows["split_mode"].str.extract(r"k_(\d+)").astype(int)
        oracle_idx = k_rows.groupby("kv_len")["ms"].idxmin()
        oracle_rows = k_rows.loc[oracle_idx].set_index("kv_len")

        common = base.index.intersection(patched.index).intersection(oracle_rows.index)
        for kv in sorted(common):
            default_ms = float(base.loc[kv, "ms"])
            patched_ms = float(patched.loc[kv, "ms"])
            default_chunks = float(base.loc[kv].get("num_chunks_kv", np.nan))
            patched_chunks = float(patched.loc[kv].get("num_chunks_kv", np.nan))

            oracle_ms = float(oracle_rows.loc[kv, "ms"])
            oracle_chunks = float(oracle_rows.loc[kv].get("num_chunks_kv", np.nan))
            oracle_k = int(oracle_rows.loc[kv, "split_k"])
            oracle_source = f"k_{oracle_k}"

            rows.append(
                {
                    "batch_size": bs,
                    "kv_len": int(kv),
                    "default_ms": default_ms,
                    "patched_ms": patched_ms,
                    "oracle_ms": oracle_ms,
                    "default_chunks": default_chunks,
                    "patched_chunks": patched_chunks,
                    "oracle_chunks": oracle_chunks,
                    "oracle_k": oracle_k,
                    "oracle_source": oracle_source,
                    "patched_speedup": default_ms / patched_ms,
                    "oracle_speedup": default_ms / oracle_ms,
                }
            )
    return pd.DataFrame(rows)


def summarize(comp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bs, grp in comp.groupby("batch_size"):
        rows.append(
            {
                "batch_size": int(bs),
                "n": len(grp),
                "patched_geomean_speedup": geomean(grp["patched_speedup"]),
                "oracle_geomean_speedup": geomean(grp["oracle_speedup"]),
                "patched_min_speedup": float(grp["patched_speedup"].min()),
                "patched_max_speedup": float(grp["patched_speedup"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("batch_size")


def plot_geomean(summary: pd.DataFrame, model: str, out_dir: Path) -> Path:
    x = np.arange(len(summary))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.bar(x - width, np.ones(len(summary)), width, label="FlashInfer default", color=GEOMEAN_DEFAULT_COLOR)
    ax.bar(x, summary["patched_geomean_speedup"], width, label="patched heuristic", color=GEOMEAN_PATCHED_COLOR)
    ax.bar(x + width, summary["oracle_geomean_speedup"], width, label="oracle", color=GEOMEAN_ORACLE_COLOR)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    for xpos, val in zip(x, summary["patched_geomean_speedup"]):
        ax.text(xpos, val + 0.005, f"{val:.3f}x", ha="center", va="bottom", fontsize=10)
    for xpos, val in zip(x + width, summary["oracle_geomean_speedup"]):
        ax.text(xpos, val + 0.005, f"{val:.3f}x", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Batch Size={int(v)}" for v in summary["batch_size"]])
    ax.set_ylabel("Geomean latency speedup vs FlashInfer default")
    ax.set_title("Patched Split-K Heuristic vs Default and Oracle")
    ax.grid(True, axis="y", alpha=0.35, linestyle=":")
    ax.legend()
    fig.tight_layout()
    out = out_dir / f"{model}_real_patched_vs_default_oracle_geomean.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_batch(comp: pd.DataFrame, model: str, bs: int, out_dir: Path) -> Path:
    grp = comp[comp["batch_size"] == bs].sort_values("kv_len")
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True)

    axes[0].plot(grp["kv_len"], grp["patched_speedup"], marker="o", ms=2.5, lw=1.3, label="patched speedup", color=PATCHED_COLOR)
    axes[0].plot(grp["kv_len"], grp["oracle_speedup"], marker="o", ms=2.5, lw=1.3, label="oracle speedup", color=ORACLE_COLOR)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="default (=1.0)")
    axes[0].set_ylabel("Speedup")
    axes[0].grid(True, alpha=0.35, linestyle=":")
    axes[0].legend()

    axes[1].plot(grp["kv_len"], grp["default_ms"], marker="o", ms=2.5, lw=1.3, label="default latency", color=DEFAULT_COLOR)
    axes[1].plot(grp["kv_len"], grp["patched_ms"], marker="o", ms=2.5, lw=1.3, label="patched latency", color=PATCHED_COLOR)
    axes[1].plot(grp["kv_len"], grp["oracle_ms"], marker="o", ms=2.5, lw=1.3, label="oracle latency", color=ORACLE_COLOR)
    axes[1].set_ylabel("Latency (ms)")
    axes[1].grid(True, alpha=0.35, linestyle=":")
    axes[1].legend()

    axes[2].plot(grp["kv_len"], grp["default_chunks"], marker="o", ms=2.5, lw=1.3, label="default # of chunks", color=DEFAULT_COLOR)
    axes[2].plot(grp["kv_len"], grp["patched_chunks"], marker="o", ms=2.5, lw=1.3, label="patched # of chunks", color=PATCHED_COLOR)
    axes[2].plot(grp["kv_len"], grp["oracle_chunks"], marker="o", ms=2.5, lw=1.3, label="oracle # of chunks", color=ORACLE_COLOR)
    axes[2].set_ylabel("# of chunks")
    axes[2].set_xlabel("kv_len")
    axes[2].grid(True, alpha=0.35, linestyle=":")
    axes[2].legend()

    fig.suptitle(f"{model} BS={bs} Real Patched Split-k Heuristic")
    fig.tight_layout()
    out = out_dir / f"{model}_bs{bs}_real_patched_detail.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare real patched split-auto results.")
    parser.add_argument("--default-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--patched-csv", type=Path, default=PATCHED_CSV)
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--batches", default="1 2 4 8 16")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    batches = [int(x) for x in args.batches.replace(",", " ").split()]
    default_df = prepare_default(pd.read_csv(args.default_csv), args.model)
    patched_df = pd.read_csv(args.patched_csv)
    comp = build_comparison(default_df, patched_df, args.model, batches)
    summary = summarize(comp)

    comp_path = args.out_dir.parent / "data" / f"{args.model}_real_patched_comparison_details.csv"
    summary_path = args.out_dir.parent / "data" / f"{args.model}_real_patched_comparison_summary.csv"
    comp.to_csv(comp_path, index=False)
    summary.to_csv(summary_path, index=False)
    geomean_plot = plot_geomean(summary, args.model, args.out_dir)
    detail_plots = [plot_batch(comp, args.model, bs, args.out_dir) for bs in batches]

    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"saved details: {comp_path}")
    print(f"saved summary: {summary_path}")
    print(f"saved geomean plot: {geomean_plot}")
    print("saved detail plots:")
    for p in detail_plots:
        print(p)


if __name__ == "__main__":
    main()
