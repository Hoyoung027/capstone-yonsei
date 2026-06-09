"""
Plot presentation-ready oracle split-K chunks, latency, and speedup by batch.

For each KV length, the oracle selects the lowest-latency candidate among the
requested split-K modes and NUM_MMA_KV settings. The FlashInfer comparison
baseline is always the baseline_before measurement for split-auto with
NUM_MMA_KV auto.

Input:
    results/data/decode_tc_results_fp16.csv

Output:
    results/plots/presentation/<model>_oracle_chunks_latency_speedup_by_batch_baseline_before.png

Usage:
    cd /root/capstone-yonsei/decode_tensor_core_experiment
    /root/capstone-yonsei/venv/bin/python \
      plots/oracle_chunks_latency_speedup_grid_plot.py \
      --model llama3_8b --all-batches --mma-candidates "auto 1 2"
"""

import argparse
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from chunk_count_plot import (
    CSV_PATH,
    available_batch_sizes,
    available_mma_candidates,
    available_split_modes,
    build_oracle_plot_data,
    load_results,
    parse_int_list,
    parse_str_list,
    split_label,
)


ROOT = pathlib.Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "results" / "plots" / "presentation"
BASELINE_MODE = "before"


def format_kv_axis(ax, max_kv_len: int) -> None:
    ax.set_xlim(left=0, right=max_kv_len + 256)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2048))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x / 1024)}K" if x >= 1024 else "")
    )
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1024))


def plot_oracle_grid(results_by_batch: dict[int, tuple], model: str) -> pathlib.Path:
    batch_sizes = list(results_by_batch)
    nrows = len(batch_sizes)
    max_kv_len = max(int(base["kv_len"].max()) for base, _ in results_by_batch.values())

    fig, axes = plt.subplots(
        nrows,
        3,
        figsize=(18.6, 9.78),
        sharex="col",
        squeeze=False,
        gridspec_kw={"wspace": 0.20, "hspace": 0.10},
    )

    for row, batch_size in enumerate(batch_sizes):
        base_frame, oracle = results_by_batch[batch_size]
        ax_chunks, ax_latency, ax_speedup = axes[row]
        default_chunks = base_frame["default_num_chunks_kv"].fillna(1)
        oracle_chunks = oracle["num_chunks_kv"].fillna(1)

        ax_chunks.plot(
            base_frame["kv_len"],
            default_chunks,
            color="#333333",
            ls="--",
            lw=1.30,
            label="FlashInfer Default",
        )
        ax_chunks.plot(
            oracle["kv_len"],
            oracle_chunks,
            color="#2CA02C",
            lw=1.70,
            label="Oracle",
        )
        max_chunks = max(
            float(default_chunks.max()),
            float(oracle_chunks.max()),
        )
        ax_chunks.set_ylim(0, max(2, math.ceil(max_chunks * 1.12)))
        ax_chunks.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))

        ax_latency.plot(
            base_frame["kv_len"],
            base_frame["default_ms"],
            color="#333333",
            ls="--",
            lw=1.30,
            label="FlashInfer Default",
        )
        ax_latency.plot(
            oracle["kv_len"],
            oracle["ms"],
            color="#D62728",
            lw=1.70,
            label="Oracle",
        )
        if batch_size == 1:
            ymin, ymax = ax_latency.get_ylim()
            ax_latency.set_ylim(ymin, max(ymax, 0.070))

        ax_speedup.axhline(1.0, color="#333333", ls="--", lw=1.15, label="Default (=1.0)")
        ax_speedup.plot(
            oracle["kv_len"],
            oracle["speedup"],
            color="#1F77B4",
            lw=1.70,
            label="Oracle Speedup",
        )
        speedup_min = min(0.99, float(oracle["speedup"].min()) - 0.005)
        speedup_max = max(1.20, float(oracle["speedup"].max()) + 0.01)
        ax_speedup.set_ylim(speedup_min, speedup_max)

        for ax in (ax_chunks, ax_latency, ax_speedup):
            format_kv_axis(ax, max_kv_len)
            ax.grid(True, which="both", ls=":", alpha=0.40)
            ax.tick_params(axis="x", labelbottom=(row == nrows - 1))

    axes[0, 0].legend(loc="upper right", fontsize=7.5, frameon=True)
    axes[0, 1].legend(loc="upper right", fontsize=7.5, frameon=True)
    axes[0, 2].legend(loc="upper right", fontsize=7.5, frameon=True)

    for ax in axes[-1]:
        ax.set_xlabel("KV Length")

    middle_row = nrows // 2
    axes[middle_row, 0].set_ylabel("# of Split-K Chunks", fontsize=10, labelpad=4)
    axes[middle_row, 1].set_ylabel("Latency (ms)", fontsize=10, labelpad=4)
    axes[middle_row, 2].set_ylabel("Speedup", fontsize=10, labelpad=4)
    fig.suptitle(
        f"{model} Decode Oracle Configuration by Batch Size",
        y=0.995,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.008,
        "Oracle selects the lowest-latency split-K / NUM_MMA_KV candidate at each KV length; "
        "FlashInfer Default uses baseline_before.",
        ha="center",
        fontsize=8,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_oracle_chunks_latency_speedup_by_batch_baseline_before.png"
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.07, top=0.94, wspace=0.20, hspace=0.10)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot stacked oracle chunks, latency, and speedup panels.")
    parser.add_argument("--csv", type=pathlib.Path, default=CSV_PATH)
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batches", default=None, help="Batch sizes to plot, e.g. '1 2 4 8 16'.")
    parser.add_argument("--all-batches", action="store_true", help="Plot every available batch size.")
    parser.add_argument(
        "--split-modes",
        default=None,
        help="Oracle split-K candidates. The off mode is excluded; default: k_1..k_20.",
    )
    parser.add_argument(
        "--mma-candidates",
        default=None,
        help="Oracle NUM_MMA_KV candidates, e.g. 'auto 1 2'. Default: every available candidate.",
    )
    parser.add_argument(
        "--include-default-candidate",
        action="store_true",
        help="Allow split-auto with the selected MMA candidates to be chosen by the oracle.",
    )
    args = parser.parse_args()

    df = load_results(args.csv)
    if args.model not in set(df["base_model"]):
        available_models = sorted(m for m in df["base_model"].unique() if m and m != "unknown")
        raise SystemExit(f"model not found: {args.model}. available: {', '.join(available_models)}")

    if args.all_batches:
        batch_sizes = available_batch_sizes(df, args.model)
    elif args.batch_size is not None:
        batch_sizes = [args.batch_size]
    else:
        batch_sizes = parse_int_list(args.batches) or available_batch_sizes(df, args.model)

    split_filter = parse_str_list(args.split_modes)
    mma_filter = parse_str_list(args.mma_candidates)
    results_by_batch = {}

    for batch_size in batch_sizes:
        split_modes = [
            mode
            for mode in (
                split_filter
                or available_split_modes(df, args.model, batch_size)
            )
            if mode not in {"auto", "off"}
        ]
        mma_candidates = mma_filter or available_mma_candidates(df, args.model, batch_size)
        base_frame, oracle = build_oracle_plot_data(
            df,
            args.model,
            batch_size,
            BASELINE_MODE,
            split_modes,
            mma_candidates,
            include_default_candidate=args.include_default_candidate,
        )
        results_by_batch[batch_size] = (base_frame, oracle)

        print(f"\n[batch={batch_size}] oracle split mode counts:")
        print(oracle["split_mode"].map(split_label).value_counts().to_string())
        print(
            f"[batch={batch_size}] oracle speedup geo/max: "
            f"{math.exp(oracle['speedup'].map(math.log).mean()):.4f} / "
            f"{oracle['speedup'].max():.4f}"
        )

    out = plot_oracle_grid(results_by_batch, args.model)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
