"""
Plot oracle best split-k speedup over split-auto baseline for tensor-core decode.

For each batch size and kv_len, this script compares all selected split-k modes
under a fixed NUM_MMA_KV setting and picks the lowest latency. The plot shows:

    speedup = split_auto_ms / min_split_mode_ms

Input:
    results/data/decode_tc_results_fp16.csv

Output:
    results/plots/max_splitk/<model>_mma_<mma>_max_splitk_by_batch.png
    results/plots/max_splitk/<model>_bs<batch>_mma_<mma>_max_splitk.png

Usage:
    cd /root/capstone-yonsei/decode_tensor_core_experiment
    /root/capstone-yonsei/venv/bin/python max_splitk_plot.py --model llama3_8b --mma auto
    /root/capstone-yonsei/venv/bin/python max_splitk_plot.py --model llama3_8b --batch-size 16 --mma 2
    /root/capstone-yonsei/venv/bin/python max_splitk_plot.py --model llama3_8b --mma auto --split-modes "off fixed_512tok fixed_1024tok fixed_2048tok fixed_4096tok fixed_8192tok"
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
PLOTS_DIR = ROOT / "results" / "plots" / "max_splitk"

BATCH_STYLES = {
    1: dict(color="#4C78A8", marker="o"),
    2: dict(color="#F58518", marker="s"),
    4: dict(color="#54A24B", marker="^"),
    8: dict(color="#B279A2", marker="D"),
    16: dict(color="#E45756", marker="v"),
}

SPLIT_ORDER = [
    "auto",
    "off",
    "fixed_128tok",
    "fixed_256tok",
    "fixed_512tok",
    "fixed_1024tok",
    "fixed_2048tok",
    "fixed_4096tok",
    "fixed_8192tok",
]


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


def mma_label(mma: str) -> str:
    return "NUM_MMA_KV auto" if mma == "auto" else f"NUM_MMA_KV={mma}"


def mma_file_label(mma: str) -> str:
    return "auto" if mma == "auto" else str(mma)


def format_kv_axis(ax, max_kv_len: int) -> None:
    ax.set_xscale("linear")
    ax.set_xlim(left=0, right=max_kv_len + 256)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1024))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}" if x >= 1024 else ""))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(128))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


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
    for col in ["kv_len", "batch_size", "forced_mma", "NUM_MMA_KV", "ms", "kv_chunk_size_tokens"]:
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


def oracle_best_for_batch(
    df: pd.DataFrame,
    model: str,
    batch_size: int,
    mma: str,
    baseline_mode: str,
    split_modes: list[str],
    include_auto_candidate: bool,
) -> pd.DataFrame:
    base = selected_mma_series(df, model, batch_size, resolve_base_split_mode(df, model, batch_size), mma, baseline_mode)
    if base.empty:
        raise SystemExit(f"baseline split mode not found for model={model}, batch={batch_size}, mma={mma}")

    candidates = []
    for split_mode in split_modes:
        if split_mode == "auto" and not include_auto_candidate:
            continue
        series = selected_mma_series(df, model, batch_size, split_mode, mma, baseline_mode)
        common = base.index.intersection(series.index)
        if common.empty:
            continue
        candidates.append(pd.DataFrame({"kv_len": common, "split_mode": split_mode, "ms": series.loc[common].values}))

    if not candidates:
        raise SystemExit(f"no split-k candidates for model={model}, batch={batch_size}, mma={mma}")

    cand = pd.concat(candidates, ignore_index=True)
    idx = cand.groupby("kv_len")["ms"].idxmin()
    best = cand.loc[idx].sort_values("kv_len").set_index("kv_len")
    common = base.index.intersection(best.index)
    out = pd.DataFrame({
        "kv_len": common,
        "baseline_ms": base.loc[common].values,
        "best_ms": best.loc[common, "ms"].values,
        "best_split_mode": best.loc[common, "split_mode"].values,
    })
    out["speedup"] = out["baseline_ms"] / out["best_ms"]
    return out


def best_mode_summary(best_df: pd.DataFrame) -> str:
    counts = best_df["best_split_mode"].value_counts().sort_index()
    return ", ".join(f"{split_label(mode)}:{count}" for mode, count in counts.items())


def plot_single_batch(best_df: pd.DataFrame, model: str, batch_size: int, mma: str) -> pathlib.Path:
    max_kv_len = int(best_df["kv_len"].max())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(1.0, color="black", lw=1.8, ls="--", label="baseline split mode (=1.0)")
    style = {"marker": "o", **BATCH_STYLES.get(batch_size, {})}
    ax.plot(best_df["kv_len"], best_df["speedup"], lw=1.55, ms=3.0, label=f"oracle best split-k", **style)

    ax.set_title(f"{model} BS={batch_size} Oracle Best Split-k Speedup ({mma_label(mma)})")
    ax.set_xlabel("kv_len")
    ax.set_ylabel("Latency Speedup")
    format_kv_axis(ax, max_kv_len)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(
        0.02,
        0.02,
        f"best mode counts: {best_mode_summary(best_df)}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9),
    )
    fig.text(
        0.5,
        0.015,
        "For each kv_len, choose the split-k setting with the lowest latency; speedup is vs baseline split mode.",
        ha="center",
        fontsize=8,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_bs{batch_size}_mma_{mma_file_label(mma)}_max_splitk.png"
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_batches(best_by_batch: dict[int, pd.DataFrame], model: str, mma: str) -> pathlib.Path:
    max_kv_len = max(int(df["kv_len"].max()) for df in best_by_batch.values())
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axhline(1.0, color="black", lw=1.8, ls="--", label="baseline split mode (=1.0)")

    for batch_size, best_df in best_by_batch.items():
        style = {"marker": "o", **BATCH_STYLES.get(batch_size, {})}
        ax.plot(
            best_df["kv_len"],
            best_df["speedup"],
            lw=1.45,
            ms=2.7,
            label=f"batch={batch_size}",
            **style,
        )

    ax.set_title(f"{model} Oracle Best Split-k Speedup by Batch ({mma_label(mma)})")
    ax.set_xlabel("kv_len")
    ax.set_ylabel("Latency Speedup")
    format_kv_axis(ax, max_kv_len)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(loc="lower right", fontsize=9)
    fig.text(
        0.5,
        0.015,
        "For each batch and kv_len, choose the split-k setting with the lowest latency; speedup is vs baseline split mode.",
        ha="center",
        fontsize=8,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_mma_{mma_file_label(mma)}_max_splitk_by_batch.png"
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot oracle best split-k speedup over split-auto baseline.")
    parser.add_argument("--csv", type=pathlib.Path, default=CSV_PATH)
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batches", default=None, help="Space/comma-separated batch sizes. Default: all available.")
    parser.add_argument("--mma", choices=["auto", "1", "2"], default="auto")
    parser.add_argument("--baseline", choices=["mean", "before", "after"], default="mean")
    parser.add_argument("--split-modes", default=None, help="Candidate split modes. Default: all non-auto modes.")
    parser.add_argument(
        "--include-auto-candidate",
        action="store_true",
        help="Allow split auto to be selected as an oracle candidate. Default excludes it because it is the baseline.",
    )
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
    best_by_batch = {}
    for batch_size in batch_sizes:
        split_modes = split_filter or [m for m in available_split_modes(df, args.model, batch_size) if m != "auto"]
        best_by_batch[batch_size] = oracle_best_for_batch(
            df,
            args.model,
            batch_size,
            args.mma,
            args.baseline,
            split_modes,
            include_auto_candidate=args.include_auto_candidate,
        )
        print(f"\n[batch={batch_size}] best split-k mode counts")
        print(best_by_batch[batch_size]["best_split_mode"].value_counts().to_string())
        print(
            f"speedup geo/mean/min/max: "
            f"{math.exp(best_by_batch[batch_size]['speedup'].map(math.log).mean()):.4f} / "
            f"{best_by_batch[batch_size]['speedup'].mean():.4f} / "
            f"{best_by_batch[batch_size]['speedup'].min():.4f} / "
            f"{best_by_batch[batch_size]['speedup'].max():.4f}"
        )

    if len(batch_sizes) == 1:
        out = plot_single_batch(best_by_batch[batch_sizes[0]], args.model, batch_sizes[0], args.mma)
    else:
        out = plot_batches(best_by_batch, args.model, args.mma)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
