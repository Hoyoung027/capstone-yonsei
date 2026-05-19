"""
Plot oracle split-k behavior for tensor-core decode.

For each batch size, this script chooses the lowest-latency candidate at every
kv_len across the selected split-k modes and NUM_MMA_KV settings. The graph
shows latency speedup, oracle latency, the selected NUM_MMA_KV, and the
num_chunks_kv for both the oracle and FlashInfer default.

Input:
    results/data/decode_tc_results_fp16.csv

Output:
    results/plots/split_k_chunks/<model>_bs<batch>_oracle_splitk_chunks.png

Usage:
    cd /root/capstone-yonsei/decode_tensor_core_experiment
    /root/capstone-yonsei/venv/bin/python plots/chunk_count_plot.py --model llama3_8b --all-batches
    /root/capstone-yonsei/venv/bin/python plots/chunk_count_plot.py --model llama3_8b --batch-size 8 --mma-candidates "auto 1 2"
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
PLOTS_DIR = ROOT / "results" / "plots" / "split_k_chunks"


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


def aggregate_metric(condition_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    before = condition_df[condition_df["phase"] == "baseline_before"].copy()
    after = condition_df[condition_df["phase"] == "baseline_after"].copy()
    if mode == "before" or after.empty:
        return before.sort_values("kv_len")
    if mode == "after" or before.empty:
        return after.sort_values("kv_len")

    cols = ["ms", "NUM_MMA_KV", "num_chunks_kv", "kv_chunk_size_tokens", "kv_chunk_size_pages", "split_kv"]
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
    values = df[(df["base_model"] == model) & (df["split_mode"] != "none")]["condition_batch_size"]
    return sorted(int(x) for x in values.dropna().unique())


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


def resolve_base_split_mode(df: pd.DataFrame, model: str, batch_size: int, preferred: str = "auto") -> str:
    modes = available_split_modes(df, model, batch_size)
    if preferred in modes:
        return preferred
    if "k_1" in modes:
        return "k_1"
    if modes:
        return modes[0]
    raise SystemExit(f"no split modes for model={model}, batch={batch_size}")


def build_oracle_plot_data(
    df: pd.DataFrame,
    model: str,
    batch_size: int,
    baseline_mode: str,
    split_modes: list[str],
    mma_candidates: list[str],
    include_default_candidate: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_split_mode = resolve_base_split_mode(df, model, batch_size, preferred="auto")
    base_frame = selected_mma_frame(df, model, batch_size, base_split_mode, "auto", baseline_mode)
    if base_frame.empty:
        raise SystemExit(f"FlashInfer default baseline not found for model={model}, batch={batch_size}")

    base_frame = base_frame.sort_values("kv_len").copy()
    base_frame["default_ms"] = base_frame["ms"]
    base_frame["default_num_chunks_kv"] = base_frame["num_chunks_kv"]
    base_frame["default_NUM_MMA_KV"] = base_frame["NUM_MMA_KV"]
    base_indexed = base_frame.set_index("kv_len")

    candidate_rows = []
    for split_mode in split_modes:
        if split_mode == base_split_mode and not include_default_candidate:
            continue
        for mma in mma_candidates:
            frame = selected_mma_frame(df, model, batch_size, split_mode, mma, baseline_mode)
            if frame.empty:
                continue
            frame = frame.set_index("kv_len")
            common = base_indexed.index.intersection(frame.index)
            if common.empty:
                continue

            out = frame.loc[common].copy()
            out["split_mode"] = split_mode
            out["mma_choice"] = mma
            out["default_ms"] = base_indexed.loc[common, "default_ms"]
            out["default_num_chunks_kv"] = base_indexed.loc[common, "default_num_chunks_kv"]
            out["default_NUM_MMA_KV"] = base_indexed.loc[common, "default_NUM_MMA_KV"]
            out["speedup"] = out["default_ms"] / out["ms"]
            candidate_rows.append(
                out.reset_index()[
                    [
                        "kv_len",
                        "split_mode",
                        "mma_choice",
                        "ms",
                        "speedup",
                        "NUM_MMA_KV",
                        "num_chunks_kv",
                        "default_ms",
                        "default_NUM_MMA_KV",
                        "default_num_chunks_kv",
                    ]
                ]
            )

    if not candidate_rows:
        raise SystemExit(f"no oracle candidates for model={model}, batch={batch_size}")

    candidates = pd.concat(candidate_rows, ignore_index=True)
    idx = candidates.groupby("kv_len")["ms"].idxmin()
    oracle = candidates.loc[idx].sort_values("kv_len").copy()
    # split-k off has no KV split chunks, so the benchmark records the chunk count as NaN.
    # Plot it at 0 to keep the oracle chunk-count line continuous and explicit.
    oracle["plot_num_chunks_kv"] = oracle["num_chunks_kv"].fillna(0)
    base_frame["plot_default_num_chunks_kv"] = base_frame["default_num_chunks_kv"].fillna(0)
    return base_frame, oracle


def plot_oracle_splitk_chunks(
    base_frame: pd.DataFrame,
    oracle: pd.DataFrame,
    model: str,
    batch_size: int,
) -> pathlib.Path:
    max_kv_len = int(base_frame["kv_len"].max())
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12.0, 10.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.15, 0.95, 1.15]},
    )
    ax_speedup, ax_latency, ax_mma, ax_chunks = axes

    ax_speedup.plot(oracle["kv_len"], oracle["speedup"], color="#1F77B4", marker="o", lw=1.8, ms=3.0, label="oracle speedup")
    ax_speedup.axhline(1.0, color="#333333", lw=1.2, ls="--", label="FlashInfer default (=1.0)")

    ax_latency.plot(oracle["kv_len"], oracle["ms"], color="#D62728", marker="o", lw=1.7, ms=2.8, label="oracle latency")
    ax_latency.plot(base_frame["kv_len"], base_frame["default_ms"], color="#333333", marker="o", ls="--", lw=1.2, ms=2.4, label="FlashInfer default latency")

    ax_mma.plot(oracle["kv_len"], oracle["NUM_MMA_KV"], color="#9467BD", marker="o", lw=1.5, ms=2.8, label="oracle NUM_MMA_KV")
    ax_mma.plot(base_frame["kv_len"], base_frame["default_NUM_MMA_KV"], color="#555555", marker="o", ls="--", lw=1.1, ms=2.2, label="FlashInfer default NUM_MMA_KV")

    ax_chunks.plot(oracle["kv_len"], oracle["plot_num_chunks_kv"], color="#2CA02C", marker="o", lw=1.7, ms=2.8, label="oracle num_chunks_kv")
    ax_chunks.plot(base_frame["kv_len"], base_frame["plot_default_num_chunks_kv"], color="#333333", marker="o", ls="--", lw=1.2, ms=2.4, label="FlashInfer default num_chunks_kv")

    ax_speedup.set_title(f"{model} BS={batch_size} Oracle Split-k Summary")
    ax_speedup.set_ylabel("Speedup")
    ax_latency.set_ylabel("Latency (ms)")
    ax_mma.set_ylabel("NUM_MMA_KV")
    ax_chunks.set_ylabel("num_chunks_kv")
    ax_chunks.set_xlabel("kv_len")
    format_kv_axis(ax_chunks, max_kv_len)

    ax_mma.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_chunks.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for ax in axes:
        ax.grid(True, which="both", ls=":", alpha=0.45)
        ax.legend(loc="best", fontsize=8, frameon=True)

    fig.text(
        0.5,
        0.015,
        "Oracle picks the lowest-latency split-k/NUM_MMA_KV candidate at each kv_len. split-k off is plotted as num_chunks_kv=0.",
        ha="center",
        fontsize=8,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_bs{batch_size}_oracle_splitk_chunks.png"
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot oracle split-k speedup and chunk counts.")
    parser.add_argument("--csv", type=pathlib.Path, default=CSV_PATH)
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", default=None, help="Batch sizes to plot, e.g. '1 2 4 8 16'.")
    parser.add_argument("--all-batches", action="store_true", help="Plot every available batch size for the selected model.")
    parser.add_argument("--baseline", choices=["mean", "before", "after"], default="mean")
    parser.add_argument("--split-modes", default=None, help="Candidate split modes. Default: all available modes except auto.")
    parser.add_argument("--mma-candidates", default=None, help="Candidate NUM_MMA_KV values. Default: all available values, e.g. 'auto 1 2'.")
    parser.add_argument("--include-default-candidate", action="store_true", help="Include FlashInfer split-auto default itself as an oracle candidate.")
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
        mma_candidates = parse_str_list(args.mma_candidates) or available_mma_candidates(df, args.model, batch_size)
        base_frame, oracle = build_oracle_plot_data(
            df,
            args.model,
            batch_size,
            args.baseline,
            split_modes,
            mma_candidates,
            include_default_candidate=args.include_default_candidate,
        )
        out = plot_oracle_splitk_chunks(base_frame, oracle, args.model, batch_size)
        saved.append(out)

        print(f"\n[batch={batch_size}] oracle split mode counts:")
        print(oracle["split_mode"].map(split_label).value_counts().to_string())
        print(f"[batch={batch_size}] oracle NUM_MMA_KV counts:")
        print(oracle["NUM_MMA_KV"].astype(int).value_counts().sort_index().to_string())
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
