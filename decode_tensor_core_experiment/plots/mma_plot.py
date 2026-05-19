"""
Plot tensor-core decode NUM_MMA_KV speedup for a selected model, batch size, and split-k mode.

Input:
    results/data/decode_tc_results_fp16.csv

Output:
    results/plots/<model>_bs<batch>_split_<split>_decode_tc_mma_speedup.png
    results/plots/<model>_bs<batch>_split_<split>_decode_tc_mma_latency.png

Usage:
    cd /root/capstone-yonsei/decode_tensor_core_experiment
    /root/capstone-yonsei/venv/bin/python mma_plot.py --model llama3_8b --batch-size 16 --split auto
    /root/capstone-yonsei/venv/bin/python mma_plot.py --model llama3_8b --batch-size 16 --split fixed_1024tok
"""

import argparse
import pathlib
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "data" / "decode_tc_results_fp16.csv"
PLOTS_DIR = ROOT / "results" / "plots" / "mma"

MMA_STYLES = {
    1: dict(color="#2e86c1", marker="o"),
    2: dict(color="#27ae60", marker="s"),
    3: dict(color="#8e44ad", marker="v"),
    4: dict(color="#e67e22", marker="^"),
    8: dict(color="#c0392b", marker="D"),
}

KV_TICKS = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]


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
    for col in [
        "kv_len",
        "batch_size",
        "CTA_TILE_Q",
        "CTA_TILE_KV",
        "NUM_MMA_KV",
        "ms",
        "tflops",
        "kv_chunk_size_tokens",
        "num_chunks_kv",
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
    return ((before.loc[common] + after.loc[common]) / 2).rename("auto_baseline_ms")


def select_condition(df: pd.DataFrame, model: str, batch_size: int, split_mode: str) -> pd.DataFrame:
    selected = df[
        (df["base_model"] == model)
        & (df["condition_batch_size"] == batch_size)
        & (df["split_mode"] == split_mode)
    ].copy()
    if selected.empty:
        available = (
            df[["base_model", "condition_batch_size", "split_mode"]]
            .drop_duplicates()
            .sort_values(["base_model", "condition_batch_size", "split_mode"])
        )
        raise SystemExit(
            f"no data for model={model}, batch_size={batch_size}, split={split_mode}\n"
            f"available:\n{available.to_string(index=False)}"
        )
    return selected


def mma_speedup_series(condition_df: pd.DataFrame, baseline_mode: str):
    base = baseline_series(condition_df, baseline_mode)
    if base.empty:
        raise SystemExit("auto baseline rows not found for selected condition")

    exps = condition_df[condition_df["phase"] == "experiment"]
    for mma, grp in sorted(exps.groupby("forced_mma"), key=lambda item: int(item[0])):
        mma = int(mma)
        g = grp.sort_values("kv_len").set_index("kv_len")
        common = base.index.intersection(g.index)
        if common.empty:
            continue
        yield mma, base.loc[common] / g.loc[common, "ms"], g.loc[common]


def plot_speedup(
    condition_df: pd.DataFrame,
    model: str,
    batch_size: int,
    split_mode: str,
    baseline_mode: str,
) -> pathlib.Path:
    max_kv_len = int(condition_df["kv_len"].max())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(1.0, color="black", lw=1.8, ls="--", label="NUM_MMA_KV auto baseline (=1.0)")

    for mma, speedup, rows in mma_speedup_series(condition_df, baseline_mode):
        style = {"marker": "o", **MMA_STYLES.get(mma, {})}
        actual_vals = sorted(rows["NUM_MMA_KV"].dropna().astype(int).unique().tolist())
        actual_text = actual_vals[0] if len(actual_vals) == 1 else ",".join(map(str, actual_vals))
        ax.plot(
            speedup.index,
            speedup.values,
            lw=1.45,
            ms=3.2,
            label=f"NUM_MMA_KV={actual_text}",
            **style,
        )

    title_split = split_mode.replace("_", " ")
    ax.set_title(f"{model} BS={batch_size} Split={title_split} Tensor-Core Decode KV-MMA Tile Speedup")
    ax.set_xlabel("kv_len")
    ax.set_ylabel("Latency Speedup")
    format_kv_axis(ax, max_kv_len)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(loc="lower right", fontsize=9)
    fig.text(
        0.5,
        0.015,
        "NUM_MMA_KV: number of tensor-core MMA tiles along the KV dimension in each CTA tile.",
        ha="center",
        fontsize=8,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_bs{batch_size}_split_{split_mode}_decode_tc_mma_speedup.png"
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_latency(condition_df: pd.DataFrame, model: str, batch_size: int, split_mode: str) -> pathlib.Path:
    max_kv_len = int(condition_df["kv_len"].max())
    fig, ax = plt.subplots(figsize=(10, 6))

    base = baseline_series(condition_df, "mean")
    if not base.empty:
        ax.plot(base.index, base.values, lw=1.7, color="#111111", ls="--", label="NUM_MMA_KV auto")

    for mma, grp in sorted(
        condition_df[condition_df["phase"] == "experiment"].groupby("forced_mma"),
        key=lambda item: int(item[0]),
    ):
        mma = int(mma)
        g = grp.sort_values("kv_len")
        style = {"marker": "o", **MMA_STYLES.get(mma, {})}
        actual_vals = sorted(g["NUM_MMA_KV"].dropna().astype(int).unique().tolist())
        actual_text = actual_vals[0] if len(actual_vals) == 1 else ",".join(map(str, actual_vals))
        ax.plot(g["kv_len"], g["ms"], lw=1.45, ms=3.0, label=f"NUM_MMA_KV={actual_text}", **style)

    title_split = split_mode.replace("_", " ")
    ax.set_title(f"{model} BS={batch_size} Split={title_split} Tensor-Core Decode KV-MMA Tile Latency")
    ax.set_xlabel("kv_len")
    ax.set_ylabel("Latency (ms)")
    ax.set_yscale("log")
    format_kv_axis(ax, max_kv_len)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(loc="best", fontsize=9)
    fig.text(
        0.5,
        0.015,
        "NUM_MMA_KV: number of tensor-core MMA tiles along the KV dimension in each CTA tile.",
        ha="center",
        fontsize=8,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_bs{batch_size}_split_{split_mode}_decode_tc_mma_latency.png"
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tensor-core decode MMA sweep for one condition.")
    parser.add_argument("--csv", type=pathlib.Path, default=CSV_PATH)
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--split",
        default="k_1",
        help="split-k mode: k_N, auto, off, or fixed_<tokens>tok such as fixed_1024tok.",
    )
    parser.add_argument(
        "--baseline",
        choices=["mean", "before", "after"],
        default="mean",
        help="Auto baseline reference for speedup. mean uses average of before/after.",
    )
    args = parser.parse_args()

    df = load_results(args.csv)
    condition_df = select_condition(df, args.model, args.batch_size, args.split)
    speedup_path = plot_speedup(condition_df, args.model, args.batch_size, args.split, args.baseline)
    latency_path = plot_latency(condition_df, args.model, args.batch_size, args.split)
    print(f"saved: {speedup_path}")
    print(f"saved: {latency_path}")


if __name__ == "__main__":
    main()
