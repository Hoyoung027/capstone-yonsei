#!/usr/bin/env python3
"""
Simulate a split-k scheduler improvement from existing sweep CSV data.

This does not patch FlashInfer.  It asks: if FlashInfer had selected split-k
chunks using a simple minimum-work-per-chunk guard, which measured k_N row in
the CSV would it have used, and how much speedup would that imply?
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "decode_tensor_core_experiment" / "results" / "data" / "decode_tc_results_fp16.csv"
OUT_DIR = ROOT / "splitk_heuristic_simulation" / "results"


@dataclass(frozen=True)
class ParsedLabel:
    phase: str
    model: str
    split_mode: str
    batch_size: int
    mma: str


def parse_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [x for x in text.replace(",", " ").split() if x]



def param_tag(alpha: float, beta: float | None = None) -> str:
    tag = f"alpha_{alpha:g}"
    if beta is not None:
        tag += f"_beta_{beta:g}"
    return tag

def geomean(values: pd.Series) -> float:
    values = values.dropna()
    values = values[values > 0]
    if values.empty:
        return float("nan")
    return float(np.exp(np.log(values).mean()))


def parse_label(label: str) -> ParsedLabel | None:
    experiment_match = re.fullmatch(r"\[experiment\]\s+(.+)_num_mma_kv_(\d+)", label)
    if experiment_match:
        condition, mma = experiment_match.groups()
        phase = "experiment"
    else:
        baseline_match = re.fullmatch(r"\[(baseline_before|baseline_after)\]\s+(.+)", label)
        if not baseline_match:
            return None
        phase, condition = baseline_match.groups()
        mma = "auto"

    condition_match = re.fullmatch(
        r"(.+?)_(?:fp16|float16|bf16|bfloat16)_split_(auto|off|fixed_\d+(?:tok)?|k_\d+)_bs(\d+)",
        condition,
    )
    if not condition_match:
        return None
    model, split_mode, batch = condition_match.groups()
    fixed_match = re.fullmatch(r"fixed_(\d+)", split_mode)
    if fixed_match:
        split_mode = f"fixed_{fixed_match.group(1)}tok"
    return ParsedLabel(phase, model, split_mode, int(batch), mma)


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    parsed = df["label"].map(parse_label)
    keep = parsed.notna()
    df = df[keep].copy()
    parsed = parsed[keep]
    df["phase"] = parsed.map(lambda x: x.phase)
    df["base_model"] = parsed.map(lambda x: x.model)
    df["split_mode"] = parsed.map(lambda x: x.split_mode)
    df["condition_batch_size"] = parsed.map(lambda x: x.batch_size)
    df["mma"] = parsed.map(lambda x: x.mma)
    return df


def baseline_frame(df: pd.DataFrame, model: str, batch_size: int, baseline: str) -> pd.DataFrame:
    if baseline not in {"before", "after", "mean"}:
        raise ValueError(f"unknown baseline: {baseline}")
    cond = df[
        (df["base_model"] == model)
        & (df["condition_batch_size"] == batch_size)
        & (df["split_mode"] == "auto")
        & (df["mma"] == "auto")
    ]
    before = cond[cond["phase"] == "baseline_before"].set_index("kv_len")
    after = cond[cond["phase"] == "baseline_after"].set_index("kv_len")
    if baseline == "before":
        base = before
    elif baseline == "after":
        base = after
    else:
        common = before.index.intersection(after.index)
        base = before.loc[common].copy()
        numeric_cols = ["ms", "num_chunks_kv", "CTA_TILE_KV", "kv_chunk_size_tokens"]
        for col in numeric_cols:
            if col in base.columns:
                base[col] = (before.loc[common, col] + after.loc[common, col]) / 2
    if base.empty:
        raise SystemExit(f"missing split_auto {baseline} baseline for model={model}, batch={batch_size}")
    return base.reset_index()


def split_k_auto_frame(df: pd.DataFrame, model: str, batch_size: int, split_k: int, baseline: str) -> pd.DataFrame:
    mode = f"k_{split_k}"
    cond = df[
        (df["base_model"] == model)
        & (df["condition_batch_size"] == batch_size)
        & (df["split_mode"] == mode)
        & (df["mma"] == "auto")
    ]
    if baseline == "before":
        out = cond[cond["phase"] == "baseline_before"]
    elif baseline == "after":
        out = cond[cond["phase"] == "baseline_after"]
    else:
        before = cond[cond["phase"] == "baseline_before"].set_index("kv_len")
        after = cond[cond["phase"] == "baseline_after"].set_index("kv_len")
        common = before.index.intersection(after.index)
        out = before.loc[common].copy()
        for col in ["ms", "num_chunks_kv", "CTA_TILE_KV", "kv_chunk_size_tokens"]:
            if col in out.columns:
                out[col] = (before.loc[common, col] + after.loc[common, col]) / 2
        out = out.reset_index()
    return out.copy()


def available_k_values(df: pd.DataFrame, model: str, batch_size: int) -> list[int]:
    modes = df[
        (df["base_model"] == model)
        & (df["condition_batch_size"] == batch_size)
        & (df["split_mode"].str.match(r"k_\d+", na=False))
    ]["split_mode"].unique()
    vals = []
    for mode in modes:
        vals.append(int(mode.split("_", 1)[1]))
    return sorted(set(vals))


def choose_proposed_k(
    default_chunks: float,
    kv_len: int,
    cta_tile_kv: float,
    alpha: float,
    available_ks: list[int],
    batch_size: int,
    beta: float | None,
) -> tuple[int, int]:
    if math.isnan(default_chunks) or default_chunks < 1:
        default_chunks = 1
    if math.isnan(cta_tile_kv) or cta_tile_kv < 1:
        cta_tile_kv = 64

    occupancy_chunks = max(1, int(round(default_chunks)))
    work_cap = max(1, int(math.floor(kv_len / (alpha * cta_tile_kv))))
    caps = [occupancy_chunks, work_cap]
    if beta is not None and beta > 0:
        caps.append(max(1, int(math.floor(beta / batch_size))))

    target = max(1, min(caps))
    chosen = min(available_ks, key=lambda k: (abs(k - target), k))
    return occupancy_chunks, chosen


def build_comparison(
    df: pd.DataFrame,
    model: str,
    batch_size: int,
    alpha: float,
    baseline: str,
    beta: float | None,
) -> pd.DataFrame:
    base = baseline_frame(df, model, batch_size, baseline)
    available_ks = available_k_values(df, model, batch_size)
    if not available_ks:
        raise SystemExit(f"no k_N split modes for model={model}, batch={batch_size}")

    k_frames = {}
    for k in available_ks:
        frame = split_k_auto_frame(df, model, batch_size, k, baseline)
        k_frames[k] = frame.set_index("kv_len")

    rows = []
    for _, b in base.iterrows():
        kv_len = int(b["kv_len"])
        default_ms = float(b["ms"])
        default_chunks = float(b.get("num_chunks_kv", 1))
        cta_tile_kv = float(b.get("CTA_TILE_KV", 64))
        default_k_equiv, proposed_k = choose_proposed_k(
            default_chunks, kv_len, cta_tile_kv, alpha, available_ks, batch_size, beta
        )
        if proposed_k == default_k_equiv:
            proposed_ms = default_ms
            proposed_chunks = default_chunks
            proposed_source = "default"
        else:
            proposed_row = k_frames[proposed_k].loc[kv_len]
            proposed_ms = float(proposed_row["ms"])
            proposed_chunks = float(proposed_row.get("num_chunks_kv", proposed_k))
            proposed_source = f"k_{proposed_k}"

        oracle_k = 0
        oracle_ms = default_ms
        oracle_chunks = default_chunks
        oracle_source = "default"
        for k, frame in k_frames.items():
            if kv_len not in frame.index:
                continue
            ms = float(frame.loc[kv_len, "ms"])
            if ms < oracle_ms:
                oracle_ms = ms
                oracle_k = k
                oracle_chunks = float(frame.loc[kv_len].get("num_chunks_kv", k))
                oracle_source = f"k_{k}"

        rows.append(
            {
                "batch_size": batch_size,
                "kv_len": kv_len,
                "default_ms": default_ms,
                "default_chunks": default_chunks,
                "cta_tile_kv": cta_tile_kv,
                "proposed_k": proposed_k,
                "proposed_source": proposed_source,
                "proposed_ms": proposed_ms,
                "proposed_chunks": proposed_chunks,
                "oracle_k": oracle_k,
                "oracle_source": oracle_source,
                "oracle_ms": oracle_ms,
                "oracle_chunks": oracle_chunks,
                "proposed_speedup": default_ms / proposed_ms,
                "oracle_speedup": default_ms / oracle_ms,
            }
        )
    return pd.DataFrame(rows)


def summarize(comparisons: list[pd.DataFrame], alpha: float, beta: float | None) -> pd.DataFrame:
    rows = []
    for comp in comparisons:
        batch_size = int(comp["batch_size"].iloc[0])
        rows.append(
            {
                "alpha": alpha,
                "beta": beta if beta is not None else np.nan,
                "batch_size": batch_size,
                "n": len(comp),
                "proposed_geomean_speedup": geomean(comp["proposed_speedup"]),
                "oracle_geomean_speedup": geomean(comp["oracle_speedup"]),
                "proposed_mean_speedup": float(comp["proposed_speedup"].mean()),
                "oracle_mean_speedup": float(comp["oracle_speedup"].mean()),
                "proposed_min_speedup": float(comp["proposed_speedup"].min()),
                "proposed_max_speedup": float(comp["proposed_speedup"].max()),
                "oracle_max_speedup": float(comp["oracle_speedup"].max()),
            }
        )
    return pd.DataFrame(rows)


def plot_geomean(summary: pd.DataFrame, model: str, alpha: float, beta: float | None, out_dir: Path) -> Path:
    batches = summary["batch_size"].astype(int).tolist()
    x = np.arange(len(batches))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(x - width, np.ones(len(batches)), width, label="FlashInfer default", color="#8c8c8c")
    ax.bar(
        x,
        summary["proposed_geomean_speedup"],
        width,
        label=f"proposed alpha={alpha:g}" + (f", beta={beta:g}" if beta else ""),
        color="#4c78a8",
    )
    ax.bar(x + width, summary["oracle_geomean_speedup"], width, label="oracle k_1..k_20", color="#f58518")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.3)

    for xpos, val in zip(x, summary["proposed_geomean_speedup"]):
        ax.text(xpos, val + 0.006, f"{val:.3f}x", ha="center", va="bottom", fontsize=10)
    for xpos, val in zip(x + width, summary["oracle_geomean_speedup"]):
        ax.text(xpos, val + 0.006, f"{val:.3f}x", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([f"BS={b}" for b in batches])
    ax.set_ylabel("Geomean latency speedup vs FlashInfer default")
    ax.set_title(f"{model} Proposed Split-k Heuristic vs Default and Oracle")
    ax.grid(True, axis="y", alpha=0.35, linestyle=":")
    ax.legend()
    fig.text(
        0.5,
        0.02,
        "Proposed rule caps FlashInfer auto chunks so each chunk has at least alpha x CTA_TILE_KV tokens; NUM_MMA_KV is kept auto.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = out_dir / f"{model}_{param_tag(alpha, beta)}_proposed_vs_default_oracle_geomean.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_batch_details(comp: pd.DataFrame, model: str, alpha: float, beta: float | None, out_dir: Path) -> Path:
    batch_size = int(comp["batch_size"].iloc[0])
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(comp["kv_len"], comp["proposed_speedup"], marker="o", ms=2.5, lw=1.4, label="proposed")
    axes[0].plot(comp["kv_len"], comp["oracle_speedup"], marker="o", ms=2.5, lw=1.4, label="oracle")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="FlashInfer default")
    axes[0].set_ylabel("Speedup")
    axes[0].legend()
    axes[0].grid(True, alpha=0.35, linestyle=":")

    axes[1].plot(comp["kv_len"], comp["default_ms"], marker="o", ms=2.5, lw=1.2, label="default")
    axes[1].plot(comp["kv_len"], comp["proposed_ms"], marker="o", ms=2.5, lw=1.2, label="proposed")
    axes[1].plot(comp["kv_len"], comp["oracle_ms"], marker="o", ms=2.5, lw=1.2, label="oracle")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.35, linestyle=":")

    axes[2].plot(comp["kv_len"], comp["default_chunks"], marker="o", ms=2.5, lw=1.2, label="default chunks")
    axes[2].plot(comp["kv_len"], comp["proposed_chunks"], marker="o", ms=2.5, lw=1.2, label="proposed chunks")
    axes[2].plot(comp["kv_len"], comp["oracle_chunks"], marker="o", ms=2.5, lw=1.2, label="oracle chunks")
    axes[2].set_ylabel("num_chunks_kv")
    axes[2].set_xlabel("kv_len")
    axes[2].legend()
    axes[2].grid(True, alpha=0.35, linestyle=":")

    fig.suptitle(f"{model} BS={batch_size} Proposed Split-k Heuristic Detail (alpha={alpha:g})")
    fig.tight_layout()
    out = out_dir / f"{model}_bs{batch_size}_{param_tag(alpha, beta)}_proposed_detail.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate proposed split-k heuristic from CSV sweep data.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--batches", default="1 2 4 8 16")
    parser.add_argument("--baseline", choices=["before", "after", "mean"], default="before")
    parser.add_argument("--alphas", default="2 4 8 16", help="Minimum work per chunk = alpha * CTA_TILE_KV.")
    parser.add_argument("--beta", type=float, default=None, help="Optional cap: chunks <= floor(beta / batch_size).")
    parser.add_argument("--pick-alpha", type=float, default=None, help="Alpha to use for detail plots. Default: best average.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(args.csv)
    batches = [int(x) for x in parse_list(args.batches)]
    alphas = [float(x) for x in parse_list(args.alphas)]

    all_summaries = []
    comparisons_by_alpha: dict[float, list[pd.DataFrame]] = {}
    for alpha in alphas:
        comps = [build_comparison(df, args.model, b, alpha, args.baseline, args.beta) for b in batches]
        comparisons_by_alpha[alpha] = comps
        all_summaries.append(summarize(comps, alpha, args.beta))

    sweep = pd.concat(all_summaries, ignore_index=True)
    beta_suffix = "" if args.beta is None else f"_beta_{args.beta:g}"
    sweep_path = args.out_dir / f"{args.model}_heuristic_alpha_sweep{beta_suffix}.csv"
    sweep.to_csv(sweep_path, index=False)

    avg = (
        sweep.groupby("alpha", as_index=False)["proposed_geomean_speedup"]
        .mean()
        .sort_values(["proposed_geomean_speedup", "alpha"], ascending=[False, True])
    )
    chosen_alpha = args.pick_alpha if args.pick_alpha is not None else float(avg.iloc[0]["alpha"])
    chosen_summary = sweep[sweep["alpha"] == chosen_alpha].copy()
    chosen_comps = comparisons_by_alpha[chosen_alpha]
    details = pd.concat(chosen_comps, ignore_index=True)
    details_path = args.out_dir / f"{args.model}_{param_tag(chosen_alpha, args.beta)}_proposed_details.csv"
    details.to_csv(details_path, index=False)

    geomean_plot = plot_geomean(chosen_summary, args.model, chosen_alpha, args.beta, args.out_dir)
    detail_plots = [plot_batch_details(comp, args.model, chosen_alpha, args.beta, args.out_dir) for comp in chosen_comps]

    print("alpha sweep:")
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\naverage proposed geomean by alpha:")
    print(avg.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\nchosen alpha: {chosen_alpha:g}")
    print(f"saved summary csv: {sweep_path}")
    print(f"saved detail csv: {details_path}")
    print(f"saved geomean plot: {geomean_plot}")
    print("saved detail plots:")
    for path in detail_plots:
        print(path)


if __name__ == "__main__":
    main()
