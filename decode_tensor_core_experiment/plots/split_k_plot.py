"""
Plot tensor-core decode split-k speedup with NUM_MMA_KV left on auto.

Input:
    results/data/decode_tc_results_fp16.csv

Output:
    results/plots/split_k/<model>_decode_tc_split_k_mma_auto_by_batch.png
    results/plots/split_k/<model>_bs<batch>_decode_tc_split_k_mma_auto_speedup.png

Usage:
    cd /root/capstone-yonsei/decode_tensor_core_experiment
    /root/capstone-yonsei/venv/bin/python split_k_plot.py --model llama3_8b
    /root/capstone-yonsei/venv/bin/python split_k_plot.py --model llama3_8b --batch-size 16
    /root/capstone-yonsei/venv/bin/python split_k_plot.py --model llama3_8b --split-modes "auto off fixed_512tok fixed_1024tok"
"""

import argparse
import math
import pathlib
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "data" / "decode_tc_results_fp16.csv"
PLOTS_DIR = ROOT / "results" / "plots" / "split_k"

SPLIT_STYLES = {
    "auto": dict(color="#111111", marker="o", ls="-"),
    "off": dict(color="#c0392b", marker="x", ls="-"),
    "fixed_128tok": dict(color="#2e86c1", marker="^", ls="-"),
    "fixed_256tok": dict(color="#16a085", marker="s", ls="-"),
    "fixed_512tok": dict(color="#27ae60", marker="D", ls="-"),
    "fixed_1024tok": dict(color="#f39c12", marker="p", ls="-"),
    "fixed_2048tok": dict(color="#d35400", marker="h", ls="-"),
    "fixed_4096tok": dict(color="#9b59b6", marker="<", ls="-"),
    "fixed_8192tok": dict(color="#1abc9c", marker=">", ls="-"),
}

KV_TICKS = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]


def parse_str_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [x for x in text.replace(",", " ").split() if x]


def format_kv_axis(ax, max_kv_len: int) -> None:
    ax.set_xscale("linear")
    ax.set_xlim(left=0, right=max_kv_len + 256)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1024))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}" if x >= 1024 else ""))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(128))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


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
    for col in [
        "kv_len",
        "batch_size",
        "NUM_MMA_KV",
        "ms",
        "kv_chunk_size_tokens",
        "kv_chunk_size_pages",
        "num_chunks_kv",
        "split_kv",
    ]:
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
    return ((before.loc[common] + after.loc[common]) / 2).rename("split_auto_mma_auto_ms")


def select_auto_condition(df: pd.DataFrame, model: str, batch_size: int, split_mode: str) -> pd.DataFrame:
    selected = df[
        (df["base_model"] == model)
        & (df["condition_batch_size"] == batch_size)
        & (df["split_mode"] == split_mode)
        & (df["phase"].isin(["baseline_before", "baseline_after"]))
    ].copy()
    return selected


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


def split_speedup_series(df: pd.DataFrame, model: str, batch_size: int, split_mode: str, baseline_mode: str):
    base_split_mode = resolve_base_split_mode(df, model, batch_size)
    base_df = select_auto_condition(df, model, batch_size, base_split_mode)
    base = baseline_series(base_df, baseline_mode)
    if base.empty:
        raise SystemExit(
            f"baseline split mode {base_split_mode} not found for model={model}, batch={batch_size}"
        )

    cond_df = select_auto_condition(df, model, batch_size, split_mode)
    series = baseline_series(cond_df, baseline_mode)
    common = base.index.intersection(series.index)
    if common.empty:
        return None
    return base.loc[common] / series.loc[common]


def plot_batch_speedup(
    df: pd.DataFrame,
    model: str,
    batch_size: int,
    split_modes: list[str],
    baseline_mode: str,
) -> pathlib.Path:
    max_kv_len = int(df[(df["base_model"] == model) & (df["condition_batch_size"] == batch_size)]["kv_len"].max())
    fig, ax = plt.subplots(figsize=(10, 6))
    base_split_mode = resolve_base_split_mode(df, model, batch_size)
    ax.axhline(1.0, color="black", lw=1.8, ls="--", label=f"{split_label(base_split_mode)} baseline (=1.0)")

    for split_mode in split_modes:
        speedup = split_speedup_series(df, model, batch_size, split_mode, baseline_mode)
        if speedup is None:
            continue
        style = {"marker": "o", "ls": "-", **SPLIT_STYLES.get(split_mode, {})}
        ax.plot(speedup.index, speedup.values, lw=1.35, ms=2.8, label=split_label(split_mode), **style)

    ax.set_title(f"{model} BS={batch_size} Tensor-Core Decode Split-k Speedup (NUM_MMA_KV auto)")
    ax.set_xlabel("kv_len")
    ax.set_ylabel("Latency Speedup")
    format_kv_axis(ax, max_kv_len)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(loc="lower right", ncol=2, fontsize=8, frameon=True)
    fig.text(
        0.5,
        0.015,
        f"Speedup is relative to {split_label(base_split_mode)} with NUM_MMA_KV auto.",
        ha="center",
        fontsize=8,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_bs{batch_size}_decode_tc_split_k_mma_auto_speedup.png"
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_batches_overview(
    df: pd.DataFrame,
    model: str,
    batch_sizes: list[int],
    split_modes_by_batch: dict[int, list[str]],
    baseline_mode: str,
) -> pathlib.Path:
    n = len(batch_sizes)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 4.8 * nrows), squeeze=False)

    legend_handles = None
    legend_labels = None
    for ax, batch_size in zip(axes.flat, batch_sizes):
        base_split_mode = resolve_base_split_mode(df, model, batch_size)
        ax.axhline(1.0, color="black", lw=1.4, ls="--", label=f"{split_label(base_split_mode)} (=1.0)")
        max_kv_len = int(df[(df["base_model"] == model) & (df["condition_batch_size"] == batch_size)]["kv_len"].max())
        for split_mode in split_modes_by_batch[batch_size]:
            speedup = split_speedup_series(df, model, batch_size, split_mode, baseline_mode)
            if speedup is None:
                continue
            style = {"marker": "o", "ls": "-", **SPLIT_STYLES.get(split_mode, {})}
            ax.plot(speedup.index, speedup.values, lw=1.1, ms=2.0, label=split_label(split_mode), **style)
        ax.set_title(f"batch={batch_size}")
        ax.set_xlabel("kv_len")
        ax.set_ylabel("Speedup")
        format_kv_axis(ax, max_kv_len)
        ax.grid(True, which="both", ls=":", alpha=0.45)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for ax in axes.flat[n:]:
        ax.axis("off")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=5,
            fontsize=8,
            frameon=True,
            bbox_to_anchor=(0.5, 0.02),
        )
    fig.suptitle(f"{model} Tensor-Core Decode Split-k Speedup by Batch (NUM_MMA_KV auto)", fontsize=15)
    fig.text(
        0.5,
        0.055,
        "Speedup is relative to the baseline split mode at the same batch size.",
        ha="center",
        fontsize=8,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_decode_tc_split_k_mma_auto_by_batch.png"
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tensor-core decode split-k sweep with NUM_MMA_KV auto.")
    parser.add_argument("--csv", type=pathlib.Path, default=CSV_PATH)
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batches", default=None, help="Space/comma-separated batch sizes. Default: all available.")
    parser.add_argument("--split-modes", default=None, help="Space/comma-separated split modes. Default: all available.")
    parser.add_argument("--baseline", choices=["mean", "before", "after"], default="mean")
    args = parser.parse_args()

    df = load_results(args.csv)
    available_models = sorted(m for m in df["base_model"].unique() if m and m != "unknown")
    if args.model not in set(df["base_model"]):
        raise SystemExit(f"model not found: {args.model}. available: {', '.join(available_models)}")

    if args.batch_size is not None:
        batch_sizes = [args.batch_size]
    elif args.batches:
        batch_sizes = [int(x) for x in parse_str_list(args.batches)]
    else:
        batch_sizes = available_batches(df, args.model)

    split_filter = parse_str_list(args.split_modes)
    split_modes_by_batch = {}
    for batch_size in batch_sizes:
        modes = available_split_modes(df, args.model, batch_size)
        if split_filter:
            modes = [mode for mode in modes if mode in split_filter]
        if not modes:
            raise SystemExit(f"no split modes for model={args.model}, batch={batch_size}")
        split_modes_by_batch[batch_size] = modes

    if len(batch_sizes) == 1:
        out = plot_batch_speedup(df, args.model, batch_sizes[0], split_modes_by_batch[batch_sizes[0]], args.baseline)
    else:
        out = plot_batches_overview(df, args.model, batch_sizes, split_modes_by_batch, args.baseline)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
