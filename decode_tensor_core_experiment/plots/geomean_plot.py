"""
Plot geometric mean speedup for tensor-core decode split-k modes at a selected NUM_MMA_KV setting.

Input:
    results/data/decode_tc_results_fp16.csv

Output:
    results/plots/split_k_geomean/<model>_bs<batch>_mma_<mma>_decode_tc_split_k_geomean.png
    results/plots/split_k_geomean/<model>_all_batches_mma_<mma>_decode_tc_split_k_geomean.png

Usage:
    cd /root/capstone-yonsei/decode_tensor_core_experiment
    /root/capstone-yonsei/venv/bin/python geomean_plot.py --model llama3_8b --batch-size 16
    /root/capstone-yonsei/venv/bin/python geomean_plot.py --model llama3_8b --all-batches
    /root/capstone-yonsei/venv/bin/python geomean_plot.py --model llama3_8b --all-batches --mma 1
    /root/capstone-yonsei/venv/bin/python geomean_plot.py --model llama3_8b --all-batches --mma 2
    /root/capstone-yonsei/venv/bin/python geomean_plot.py --model llama3_8b --batch-size 16 --split-modes "off fixed_512tok fixed_1024tok fixed_2048tok"
"""

import argparse
import math
import pathlib
import re

import matplotlib.pyplot as plt
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "data" / "decode_tc_results_fp16.csv"
PLOTS_DIR = ROOT / "results" / "plots" / "split_k_geomean"

SPLIT_COLORS = {
    "auto": "#2F2F2F",
    "off": "#9D4D4D",
    "fixed_128tok": "#4C78A8",
    "fixed_256tok": "#72B7B2",
    "fixed_512tok": "#54A24B",
    "fixed_1024tok": "#F2A541",
    "fixed_2048tok": "#E45756",
    "fixed_4096tok": "#B279A2",
    "fixed_8192tok": "#59A14F",
}

BATCH_COLORS = {
    1: "#4C78A8",
    2: "#F58518",
    4: "#54A24B",
    8: "#B279A2",
    16: "#E45756",
}



def parse_str_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [x for x in text.replace(",", " ").split() if x]


def split_sort_key(split_mode: str) -> tuple[int, int]:
    if split_mode == "auto":
        return (0, 0)
    if split_mode == "off":
        return (1, 0)
    match = re.fullmatch(r"fixed_(\d+)tok", split_mode)
    if match:
        return (2, int(match.group(1)))
    match = re.fullmatch(r"k_(\d+)", split_mode)
    if match:
        return (3, int(match.group(1)))
    return (4, 0)


def split_label(split_mode: str) -> str:
    if split_mode == "auto":
        return "auto"
    if split_mode == "off":
        return "off"
    match = re.fullmatch(r"fixed_(\d+)tok", split_mode)
    if match:
        return f"chunk={match.group(1)}"
    match = re.fullmatch(r"k_(\d+)", split_mode)
    if match:
        return f"k={match.group(1)}"
    return split_mode


def parse_label(label: str) -> tuple[str, str, int | None]:
    baseline_match = re.fullmatch(r"\[(baseline_before|baseline_after)\]\s+(.+)", label)
    if baseline_match:
        phase, condition = baseline_match.groups()
        return condition, phase, None

    experiment_match = re.fullmatch(r"\[experiment\]\s+(.+)_num_mma_kv_(\d+)", label)
    if experiment_match:
        condition, mma = experiment_match.groups()
        return condition, "experiment", int(mma)

    return label, "unknown", None


def parse_condition(condition: str) -> tuple[str, str, int | None]:
    # Supported labels include:
    #   llama3_8b_float16_split_k_11_bs16
    #   llama3_8b_fp16_split_fixed_1024tok_bs16
    #   llama3_8b_float16_split_fixed_1024_bs16
    match = re.fullmatch(
        r"(.+?)_(?:fp16|float16|bf16|bfloat16)_split_(auto|off|fixed_\d+(?:tok)?|k_\d+)_bs(\d+).*",
        condition,
    )
    if match:
        model, split_mode, batch_size = match.groups()
        fixed_match = re.fullmatch(r"fixed_(\d+)", split_mode)
        if fixed_match:
            split_mode = f"fixed_{fixed_match.group(1)}tok"
        return model, split_mode, int(batch_size)
    return condition, "none", None


def load_results(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["kv_len", "batch_size", "NUM_MMA_KV", "ms", "kv_chunk_size_tokens", "split_kv"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    parsed = df["label"].apply(parse_label)
    df["condition"] = parsed.apply(lambda x: x[0])
    df["phase"] = parsed.apply(lambda x: x[1])
    df["forced_mma"] = parsed.apply(lambda x: x[2])

    condition_parsed = df["condition"].apply(parse_condition)
    df["base_model"] = condition_parsed.apply(lambda x: x[0])
    df["split_mode"] = condition_parsed.apply(lambda x: x[1])
    parsed_batch = condition_parsed.apply(lambda x: x[2])
    df["condition_batch_size"] = parsed_batch.combine_first(df["batch_size"])
    df["condition_batch_size"] = pd.to_numeric(df["condition_batch_size"], errors="coerce")
    return df


def baseline_series(condition_df: pd.DataFrame, mode: str) -> pd.Series:
    before = condition_df[condition_df["phase"] == "baseline_before"].sort_values("kv_len").set_index("kv_len")["ms"]
    after = condition_df[condition_df["phase"] == "baseline_after"].sort_values("kv_len").set_index("kv_len")["ms"]

    if mode == "before":
        return before
    if mode == "after":
        return after
    if before.empty:
        return after
    if after.empty:
        return before

    common = before.index.intersection(after.index)
    return ((before.loc[common] + after.loc[common]) / 2).rename("ms")


def geometric_mean(values: pd.Series) -> float:
    values = values.dropna()
    values = values[values > 0]
    if values.empty:
        return float("nan")
    return math.exp(values.map(math.log).mean())


def mma_label(mma: str) -> str:
    return "NUM_MMA_KV auto" if mma == "auto" else f"NUM_MMA_KV={mma}"


def mma_file_label(mma: str) -> str:
    return "auto" if mma == "auto" else str(mma)


def selected_mma_series(
    df: pd.DataFrame,
    model: str,
    batch_size: int,
    split_mode: str,
    mma: str,
    baseline_mode: str,
) -> pd.Series:
    cond = df[
        (df["base_model"] == model)
        & (df["condition_batch_size"] == batch_size)
        & (df["split_mode"] == split_mode)
    ].copy()

    if mma == "auto":
        return baseline_series(
            cond[cond["phase"].isin(["baseline_before", "baseline_after"])], baseline_mode
        )

    forced_mma = int(mma)
    exp = cond[(cond["phase"] == "experiment") & (cond["forced_mma"] == forced_mma)]
    return exp.sort_values("kv_len").set_index("kv_len")["ms"]


def available_batches(df: pd.DataFrame, model: str) -> list[int]:
    values = df[(df["base_model"] == model) & (df["split_mode"] != "none")]["condition_batch_size"]
    return sorted(values.dropna().astype(int).unique().tolist())




def resolve_base_split_mode(df: pd.DataFrame, model: str, batch_size: int, preferred: str = "auto") -> str:
    modes = available_split_modes(df, model, batch_size)
    if preferred in modes:
        return preferred
    if "k_1" in modes:
        return "k_1"
    if modes:
        return modes[0]
    raise SystemExit(f"no split modes for model={model}, batch={batch_size}")

def available_split_modes(df: pd.DataFrame, model: str, batch_size: int) -> list[str]:
    values = df[
        (df["base_model"] == model)
        & (df["condition_batch_size"] == batch_size)
        & (df["split_mode"] != "none")
    ]["split_mode"]
    return sorted(values.dropna().unique().tolist(), key=split_sort_key)


def available_mma_candidates(df: pd.DataFrame, model: str, batch_size: int) -> list[str]:
    cond = df[
        (df["base_model"] == model)
        & (df["condition_batch_size"] == batch_size)
        & (df["split_mode"] != "none")
    ]
    values = []
    if not cond[cond["phase"].isin(["baseline_before", "baseline_after"])].empty:
        values.append("auto")
    values.extend(str(int(x)) for x in sorted(cond["forced_mma"].dropna().unique()))
    return values


def oracle_geomean_rows(
    df: pd.DataFrame,
    model: str,
    batch_sizes: list[int],
    split_filter: list[str],
    mma_candidates_filter: list[str],
    baseline_mode: str,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    rows = []
    oracle_details = {}

    for batch_size in batch_sizes:
        base_split_mode = resolve_base_split_mode(df, model, batch_size, preferred="auto")
        base = selected_mma_series(df, model, batch_size, base_split_mode, "auto", baseline_mode)
        if base.empty:
            raise SystemExit(f"baseline not found for model={model}, batch={batch_size}")

        split_modes = split_filter or [
            mode for mode in available_split_modes(df, model, batch_size)
            if re.fullmatch(r"k_\d+", mode)
        ]
        mma_candidates = mma_candidates_filter or available_mma_candidates(df, model, batch_size)

        candidate_rows = []
        for split_mode in split_modes:
            for mma in mma_candidates:
                series = selected_mma_series(df, model, batch_size, split_mode, mma, baseline_mode)
                common = base.index.intersection(series.index)
                if common.empty:
                    continue
                frame = pd.DataFrame({
                    "kv_len": common,
                    "split_mode": split_mode,
                    "mma": mma,
                    "ms": series.loc[common].values,
                    "baseline_ms": base.loc[common].values,
                })
                frame["speedup"] = frame["baseline_ms"] / frame["ms"]
                candidate_rows.append(frame)

        if not candidate_rows:
            raise SystemExit(f"no oracle candidates for model={model}, batch={batch_size}")

        candidates = pd.concat(candidate_rows, ignore_index=True)
        idx = candidates.groupby("kv_len")["ms"].idxmin()
        oracle = candidates.loc[idx].sort_values("kv_len").copy()
        oracle_details[batch_size] = oracle

        rows.append({
            "batch_size": batch_size,
            "n": int(oracle["speedup"].count()),
            "geo_mean_speedup": geometric_mean(oracle["speedup"]),
            "arith_mean_speedup": float(oracle["speedup"].mean()),
            "min_speedup": float(oracle["speedup"].min()),
            "max_speedup": float(oracle["speedup"].max()),
            "top_split_mode": oracle["split_mode"].map(split_label).value_counts().idxmax(),
        })

    return pd.DataFrame(rows), oracle_details


def plot_oracle_geomean_by_batch(summary: pd.DataFrame, model: str, baseline_mode: str) -> pathlib.Path:
    if summary.empty:
        raise SystemExit("no oracle geomean rows to plot")

    summary = summary.sort_values("batch_size")
    labels = [f"BS={int(x)}" for x in summary["batch_size"]]
    values = summary["geo_mean_speedup"].tolist()
    colors = [BATCH_COLORS.get(int(batch), "#4C78A8") for batch in summary["batch_size"]]

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    bars = ax.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.65, alpha=0.92)
    ax.axhline(1.0, color="black", lw=1.4, ls="--", label="FlashInfer default (=1.0)")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(f"{model} Oracle Split-k Geomean Speedup by Batch")
    ax.set_xlabel("batch size")
    ax.set_ylabel("Geomean Latency Speedup")
    ax.set_ylim(bottom=0, top=max(1.08, max(values) * 1.12))
    ax.grid(True, axis="y", ls=":", alpha=0.45)
    ax.legend(loc="best", fontsize=9)
    fig.text(
        0.5,
        0.015,
        f"Oracle picks the lowest-latency k_1..k_20 / NUM_MMA_KV candidate at each kv_len; baseline={baseline_mode}, FlashInfer split-auto NUM_MMA_KV auto.",
        ha="center",
        fontsize=8,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_oracle_geomean_by_batch.png"
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def geomean_rows(
    df: pd.DataFrame,
    model: str,
    batch_size: int,
    split_modes: list[str],
    baseline_mode: str,
    mma: str,
    include_auto: bool = False,
) -> pd.DataFrame:
    base = selected_mma_series(df, model, batch_size, resolve_base_split_mode(df, model, batch_size), mma, baseline_mode)
    if base.empty:
        raise SystemExit(
            f"baseline split mode not found for model={model}, batch={batch_size}, mma={mma}"
        )

    rows = []
    for split_mode in split_modes:
        if split_mode == "auto" and not include_auto:
            continue
        series = selected_mma_series(df, model, batch_size, split_mode, mma, baseline_mode)
        common = base.index.intersection(series.index)
        if common.empty:
            continue
        speedup = base.loc[common] / series.loc[common]
        chunk_tokens = None
        if split_mode.startswith("fixed_"):
            match = re.fullmatch(r"fixed_(\d+)tok", split_mode)
            chunk_tokens = int(match.group(1)) if match else None
        rows.append({
            "split_mode": split_mode,
            "label": split_label(split_mode),
            "chunk_tokens": chunk_tokens,
            "n": int(speedup.count()),
            "geo_mean_speedup": geometric_mean(speedup),
            "arith_mean_speedup": float(speedup.mean()),
            "min_speedup": float(speedup.min()),
            "max_speedup": float(speedup.max()),
        })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["sort_key"] = out["split_mode"].apply(split_sort_key)
    return out.sort_values("sort_key").drop(columns=["sort_key"])


def plot_geomean(
    summary: pd.DataFrame, model: str, batch_size: int, baseline_mode: str, mma: str
) -> pathlib.Path:
    if summary.empty:
        raise SystemExit("no rows to plot")

    labels = summary["label"].tolist()
    values = summary["geo_mean_speedup"].tolist()
    colors = [SPLIT_COLORS.get(mode, "#4C78A8") for mode in summary["split_mode"]]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.55, alpha=0.92)
    ax.axhline(1.0, color="black", lw=1.5, ls="--", label="baseline split mode (=1.0)")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}x",
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=0,
        )

    ax.set_title(f"{model} BS={batch_size} Split-k Geomean Speedup ({mma_label(mma)})")
    ax.set_xlabel("split-k mode / fixed chunk size (tokens)")
    ax.set_ylabel("Geomean Latency Speedup")
    ax.grid(True, axis="y", ls=":", alpha=0.45)
    ax.legend(loc="best", fontsize=9)
    fig.text(
        0.5,
        0.015,
        f"Speedup = baseline_ms / split_mode_ms over kv_len points; baseline={baseline_mode}, {mma_label(mma)}.",
        ha="center",
        fontsize=8,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_bs{batch_size}_mma_{mma_file_label(mma)}_decode_tc_split_k_geomean.png"
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_multi_batch_geomean(
    summaries: dict[int, pd.DataFrame], model: str, baseline_mode: str, mma: str
) -> pathlib.Path:
    summaries = {batch: summary for batch, summary in summaries.items() if not summary.empty}
    if not summaries:
        raise SystemExit("no rows to plot")

    batch_sizes = list(summaries)
    split_modes = sorted(
        {mode for summary in summaries.values() for mode in summary["split_mode"]},
        key=split_sort_key,
    )
    labels = [split_label(mode) for mode in split_modes]
    n_batches = len(batch_sizes)
    width = 0.78 / max(n_batches, 1)
    centers = list(range(len(split_modes)))

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(split_modes) + 1.4 * n_batches), 5.8))
    ax.axhline(1.0, color="black", lw=1.5, ls="--", label="baseline split mode (=1.0)")

    for batch_idx, batch_size in enumerate(batch_sizes):
        summary = summaries[batch_size].set_index("split_mode")
        offset = (batch_idx - (n_batches - 1) / 2) * width
        xs = [center + offset for center in centers]
        ys = [summary.loc[mode, "geo_mean_speedup"] if mode in summary.index else float("nan") for mode in split_modes]
        color = BATCH_COLORS.get(batch_size, "#4C78A8")
        bars = ax.bar(
            xs,
            ys,
            width=width,
            label=f"batch={batch_size}",
            color=color,
            edgecolor="#222222",
            linewidth=0.45,
            alpha=0.92,
        )
        for bar, value in zip(bars, ys):
            if pd.isna(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}x",
                ha="center",
                va="bottom",
                fontsize=5.5,
                rotation=0,
            )

    ax.set_title(f"{model} Split-k Geomean Speedup by Batch ({mma_label(mma)})")
    ax.set_xlabel("split-k mode / fixed chunk size (tokens)")
    ax.set_ylabel("Geomean Latency Speedup")
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", ls=":", alpha=0.45)
    ax.legend(loc="best", fontsize=9)
    fig.text(
        0.5,
        0.015,
        f"Speedup = baseline_ms / split_mode_ms over kv_len points; baseline={baseline_mode}, {mma_label(mma)}.",
        ha="center",
        fontsize=8,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_all_batches_mma_{mma_file_label(mma)}_decode_tc_split_k_geomean.png"
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot split-k geometric mean speedup with NUM_MMA_KV auto.")
    parser.add_argument("--csv", type=pathlib.Path, default=CSV_PATH)
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", default=None, help="Space/comma-separated batch sizes for one combined plot.")
    parser.add_argument("--all-batches", action="store_true", help="Plot all available batch sizes in one figure.")
    parser.add_argument("--split-modes", default=None, help="Space/comma-separated split modes. Default: all available.")
    parser.add_argument("--baseline", choices=["mean", "before", "after"], default="mean")
    parser.add_argument("--mma", choices=["auto", "1", "2"], default="auto")
    parser.add_argument("--include-auto", action="store_true", help="Include split auto as a 1.0 bar; by default it is shown only as the baseline line.")
    parser.add_argument("--oracle-by-batch", action="store_true", help="Plot one oracle geomean speedup bar per batch size.")
    parser.add_argument("--mma-candidates", default=None, help="NUM_MMA_KV oracle candidates, e.g. 'auto 1 2'. Default: all available.")
    args = parser.parse_args()

    df = load_results(args.csv)
    available_models = sorted(m for m in df["base_model"].unique() if m and m != "unknown")
    if args.model not in set(df["base_model"]):
        raise SystemExit(f"model not found: {args.model}. available: {', '.join(available_models)}")

    if args.oracle_by_batch:
        batch_sizes = available_batches(df, args.model) if args.all_batches or not args.batches else [int(x) for x in parse_str_list(args.batches)]
        split_filter = parse_str_list(args.split_modes)
        mma_candidates = parse_str_list(args.mma_candidates)
        summary, details = oracle_geomean_rows(
            df,
            args.model,
            batch_sizes,
            split_filter,
            mma_candidates,
            args.baseline,
        )
        out = plot_oracle_geomean_by_batch(summary, args.model, args.baseline)
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        for batch_size, oracle in details.items():
            print(f"\n[batch={batch_size}] oracle split mode counts:")
            print(oracle["split_mode"].map(split_label).value_counts().to_string())
            print(f"[batch={batch_size}] oracle NUM_MMA_KV counts:")
            print(oracle["mma"].value_counts().sort_index().to_string())
        print(f"saved: {out}")
        return

    if args.all_batches or args.batches:
        batch_sizes = available_batches(df, args.model) if args.all_batches else [int(x) for x in parse_str_list(args.batches)]
        split_filter = parse_str_list(args.split_modes)
        summaries = {}
        for batch_size in batch_sizes:
            split_modes = split_filter or available_split_modes(df, args.model, batch_size)
            if not split_modes:
                raise SystemExit(f"no split modes for model={args.model}, batch={batch_size}")
            summaries[batch_size] = geomean_rows(
                df,
                args.model,
                batch_size,
                split_modes,
                args.baseline,
                args.mma,
                include_auto=args.include_auto,
            )

        out = plot_multi_batch_geomean(summaries, args.model, args.baseline, args.mma)
        for batch_size, summary in summaries.items():
            print(f"\n[batch={batch_size}]")
            print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        print(f"saved: {out}")
        return

    split_modes = parse_str_list(args.split_modes) or available_split_modes(df, args.model, args.batch_size)
    if not split_modes:
        raise SystemExit(f"no split modes for model={args.model}, batch={args.batch_size}")

    summary = geomean_rows(
        df,
        args.model,
        args.batch_size,
        split_modes,
        args.baseline,
        args.mma,
        include_auto=args.include_auto,
    )
    out = plot_geomean(summary, args.model, args.batch_size, args.baseline, args.mma)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
