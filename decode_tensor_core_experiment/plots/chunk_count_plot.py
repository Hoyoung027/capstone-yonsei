"""
Plot split-k speedup and actual split chunk counts for tensor-core decode.

For a selected model/batch/NUM_MMA_KV setting, this script draws two aligned panels:

Top:
    actual num_chunks_kv used by each split-k mode, plus the oracle-best mode count.
Bottom:
    speedup = split_auto_ms / split_mode_ms for each split-k mode, plus oracle-best speedup.

Input:
    results/data/decode_tc_results_fp16.csv

Output:
    results/plots/split_k_chunks/<model>_bs<batch>_mma_<mma>_splitk_chunks_speedup.png

Usage:
    cd /root/capstone-yonsei/decode_tensor_core_experiment
    /root/capstone-yonsei/venv/bin/python chunk_count_plot.py --model llama3_8b --batch-size 8 --mma auto
    /root/capstone-yonsei/venv/bin/python chunk_count_plot.py --model llama3_8b --all-batches --mma auto
    /root/capstone-yonsei/venv/bin/python chunk_count_plot.py --model llama3_8b --batch-size 8 --mma 2 --split-modes "fixed_512tok fixed_1024tok fixed_2048tok fixed_4096tok fixed_8192tok"
"""

import argparse
import math
import pathlib
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


ROOT = pathlib.Path(__file__).parent
CSV_PATH = ROOT / "results" / "data" / "decode_tc_results_fp16.csv"
PLOTS_DIR = ROOT / "results" / "plots" / "split_k_chunks"

SPLIT_STYLES = {
    "auto": dict(color="#2F2F2F", marker="o", ls="-"),
    "off": dict(color="#9D4D4D", marker="x", ls="-"),
    "fixed_128tok": dict(color="#4C78A8", marker="^", ls="-"),
    "fixed_256tok": dict(color="#72B7B2", marker="s", ls="-"),
    "fixed_512tok": dict(color="#54A24B", marker="D", ls="-"),
    "fixed_1024tok": dict(color="#F2A541", marker="p", ls="-"),
    "fixed_2048tok": dict(color="#E45756", marker="h", ls="-"),
    "fixed_4096tok": dict(color="#B279A2", marker="<", ls="-"),
    "fixed_8192tok": dict(color="#59A14F", marker=">", ls="-"),
}


def parse_str_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [x for x in text.replace(",", " ").split() if x]


def parse_int_list(text: str | None) -> list[int]:
    return [int(x) for x in parse_str_list(text)]


def split_sort_key(split_mode: str) -> tuple[int, int]:
    if split_mode == "auto":
        return (0, 0)
    if split_mode == "off":
        return (1, 0)
    match = re.fullmatch(r"fixed_(\d+)tok", split_mode)
    if match:
        return (2, int(match.group(1)))
    return (3, 0)


def split_label(split_mode: str) -> str:
    if split_mode == "auto":
        return "auto"
    if split_mode == "off":
        return "off"
    match = re.fullmatch(r"fixed_(\d+)tok", split_mode)
    if match:
        return f"chunk={match.group(1)}"
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
    match = re.fullmatch(r"(.+?)_fp16_split_(auto|off|fixed_\d+tok)_bs(\d+).*", condition)
    if match:
        model, split_mode, batch_size = match.groups()
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


def aggregate_metric(condition_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    before = condition_df[condition_df["phase"] == "baseline_before"].copy()
    after = condition_df[condition_df["phase"] == "baseline_after"].copy()
    if mode == "before" or after.empty:
        return before.sort_values("kv_len")
    if mode == "after" or before.empty:
        return after.sort_values("kv_len")

    cols = ["ms", "num_chunks_kv", "kv_chunk_size_tokens", "kv_chunk_size_pages", "split_kv"]
    merged = before[["kv_len", *cols]].merge(after[["kv_len", *cols]], on="kv_len", suffixes=("_before", "_after"))
    out = pd.DataFrame({"kv_len": merged["kv_len"]})
    for col in cols:
        out[col] = (merged[f"{col}_before"] + merged[f"{col}_after"]) / 2
    return out.sort_values("kv_len")


def selected_mma_frame(
    df: pd.DataFrame,
    model: str,
    batch_size: int,
    split_mode: str,
    mma: str,
    baseline_mode: str,
) -> pd.DataFrame:
    cond = df[
        (df["base_model"] == model)
        & (df["condition_batch_size"] == batch_size)
        & (df["split_mode"] == split_mode)
    ].copy()

    if mma == "auto":
        return aggregate_metric(cond[cond["phase"].isin(["baseline_before", "baseline_after"])], baseline_mode)

    forced_mma = int(mma)
    exp = cond[(cond["phase"] == "experiment") & (cond["forced_mma"] == forced_mma)]
    return exp.sort_values("kv_len")


def available_split_modes(df: pd.DataFrame, model: str, batch_size: int) -> list[str]:
    values = df[
        (df["base_model"] == model)
        & (df["condition_batch_size"] == batch_size)
        & (df["split_mode"] != "none")
    ]["split_mode"]
    return sorted(values.dropna().unique().tolist(), key=split_sort_key)


def available_batch_sizes(df: pd.DataFrame, model: str) -> list[int]:
    values = df[
        (df["base_model"] == model)
        & (df["split_mode"] != "none")
    ]["condition_batch_size"]
    return sorted(int(x) for x in values.dropna().unique())


def build_plot_data(
    df: pd.DataFrame,
    model: str,
    batch_size: int,
    mma: str,
    baseline_mode: str,
    split_modes: list[str],
    include_auto_candidate: bool,
) -> tuple[pd.Series, dict[str, pd.DataFrame], pd.DataFrame]:
    base_frame = selected_mma_frame(df, model, batch_size, "auto", mma, baseline_mode)
    if base_frame.empty:
        raise SystemExit(f"split auto baseline not found for model={model}, batch={batch_size}, mma={mma}")
    base_frame = base_frame.set_index("kv_len")
    base = base_frame["ms"]

    mode_frames = {}
    candidate_rows = []
    for split_mode in split_modes:
        if split_mode == "auto" and not include_auto_candidate:
            continue
        frame = selected_mma_frame(df, model, batch_size, split_mode, mma, baseline_mode)
        if frame.empty:
            continue
        frame = frame.set_index("kv_len")
        common = base.index.intersection(frame.index)
        if common.empty:
            continue
        out = frame.loc[common].copy()
        out["speedup"] = base.loc[common] / out["ms"]
        out["split_mode"] = split_mode
        mode_frames[split_mode] = out.reset_index()
        candidate_rows.append(out.reset_index()[["kv_len", "split_mode", "ms", "speedup", "num_chunks_kv"]])

    if not candidate_rows:
        raise SystemExit("no split-k candidates to plot")

    candidates = pd.concat(candidate_rows, ignore_index=True)
    idx = candidates.groupby("kv_len")["ms"].idxmin()
    oracle = candidates.loc[idx].sort_values("kv_len").copy()
    return base, base_frame.reset_index(), mode_frames, oracle


def plot_chunk_counts_and_speedup(
    base: pd.Series,
    base_frame: pd.DataFrame,
    mode_frames: dict[str, pd.DataFrame],
    oracle: pd.DataFrame,
    model: str,
    batch_size: int,
    mma: str,
) -> pathlib.Path:
    max_kv_len = int(base.index.max())
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(11.5, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.4]},
    )

    for split_mode, frame in sorted(mode_frames.items(), key=lambda item: split_sort_key(item[0])):
        style = SPLIT_STYLES.get(split_mode, {})
        label = split_label(split_mode)
        ax_top.plot(
            frame["kv_len"],
            frame["num_chunks_kv"],
            lw=1.0,
            ms=2.0,
            alpha=0.58,
            label=label,
            **style,
        )
        ax_bottom.plot(
            frame["kv_len"],
            frame["speedup"],
            lw=1.25,
            ms=2.5,
            alpha=0.85,
            label=label,
            **style,
        )

    ax_top.plot(
        base_frame["kv_len"],
        base_frame["num_chunks_kv"],
        color="black",
        lw=1.7,
        ls="--",
        marker=".",
        ms=2.4,
        label="split auto chunks",
    )

    ax_top.plot(
        oracle["kv_len"],
        oracle["num_chunks_kv"],
        color="black",
        lw=2.2,
        marker="o",
        ms=2.8,
        label="oracle best chunks",
    )
    ax_bottom.plot(
        oracle["kv_len"],
        oracle["speedup"],
        color="black",
        lw=2.3,
        marker="o",
        ms=3.0,
        label="oracle best",
    )
    ax_bottom.axhline(1.0, color="black", lw=1.5, ls="--", label="split auto baseline (=1.0)")

    ax_top.set_title(f"{model} BS={batch_size} Split-k Chunk Count and Speedup ({mma_label(mma)})")
    ax_top.set_ylabel("num_chunks_kv")
    ax_bottom.set_ylabel("Latency Speedup")
    ax_bottom.set_xlabel("kv_len")
    format_kv_axis(ax_bottom, max_kv_len)

    for ax in (ax_top, ax_bottom):
        ax.grid(True, which="both", ls=":", alpha=0.45)

    handles, labels = ax_bottom.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.02))
    fig.text(
        0.5,
        0.07,
        "Top: actual split chunk count. Bottom: speedup vs split auto; oracle picks the lowest-latency split-k mode at each kv_len.",
        ha="center",
        fontsize=8,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_bs{batch_size}_mma_{mma_file_label(mma)}_splitk_chunks_speedup.png"
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot split-k speedup with actual split chunk counts.")
    parser.add_argument("--csv", type=pathlib.Path, default=CSV_PATH)
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", default=None, help="Batch sizes to plot, e.g. '1 2 4 8 16'.")
    parser.add_argument("--all-batches", action="store_true", help="Plot every available batch size for the selected model.")
    parser.add_argument("--mma", choices=["auto", "1", "2"], default="auto")
    parser.add_argument("--baseline", choices=["mean", "before", "after"], default="mean")
    parser.add_argument("--split-modes", default=None, help="Candidate split modes. Default: all non-auto modes.")
    parser.add_argument("--include-auto-candidate", action="store_true", help="Plot/include split auto as an oracle candidate.")
    args = parser.parse_args()

    df = load_results(args.csv)
    available_models = sorted(m for m in df["base_model"].unique() if m and m != "unknown")
    if args.model not in set(df["base_model"]):
        raise SystemExit(f"model not found: {args.model}. available: {', '.join(available_models)}")

    if args.all_batches:
        batch_sizes = available_batch_sizes(df, args.model)
    else:
        batch_sizes = parse_int_list(args.batches) or [args.batch_size]

    saved = []
    for batch_size in batch_sizes:
        split_modes = parse_str_list(args.split_modes) or [
            m for m in available_split_modes(df, args.model, batch_size) if m != "auto"
        ]
        base, base_frame, mode_frames, oracle = build_plot_data(
            df,
            args.model,
            batch_size,
            args.mma,
            args.baseline,
            split_modes,
            include_auto_candidate=args.include_auto_candidate,
        )
        out = plot_chunk_counts_and_speedup(
            base, base_frame, mode_frames, oracle, args.model, batch_size, args.mma
        )
        saved.append(out)

        print(f"\n[batch={batch_size}] oracle best split-k mode counts:")
        print(oracle["split_mode"].value_counts().to_string())
        print(
            f"[batch={batch_size}] oracle speedup geo/mean/min/max: "
            f"{math.exp(oracle['speedup'].map(math.log).mean()):.4f} / "
            f"{oracle['speedup'].mean():.4f} / "
            f"{oracle['speedup'].min():.4f} / "
            f"{oracle['speedup'].max():.4f}"
        )
        print(f"[batch={batch_size}] saved: {out}")

    print("\nsaved files:")
    for out in saved:
        print(out)


if __name__ == "__main__":
    main()
