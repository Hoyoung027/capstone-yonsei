"""
Plot presentation-ready oracle latency and speedup panels by batch size.

The left column shows FlashInfer default latency and oracle latency for each
batch size. The right column shows the corresponding oracle speedup. Oracle
candidates are selected in the same way as chunk_count_plot.py: for each
kv_len, choose the lowest-latency row among the requested split-k modes and
NUM_MMA_KV candidates.

Unlike the exploratory plots, the FlashInfer comparison baseline is always
the baseline_before measurement for split-auto with NUM_MMA_KV auto.

Input:
    results/data/decode_tc_results_fp16.csv

Output:
    results/plots/presentation/<model>_oracle_latency_speedup_by_batch_baseline_before.png

Usage:
    cd /root/capstone-yonsei/decode_tensor_core_experiment
    /root/capstone-yonsei/venv/bin/python plots/oracle_latency_speedup_grid_plot.py --model llama3_8b --all-batches
    /root/capstone-yonsei/venv/bin/python plots/oracle_latency_speedup_grid_plot.py --model llama3_8b --all-batches --mma-candidates "auto 1 2"
    /root/capstone-yonsei/venv/bin/python plots/oracle_latency_speedup_grid_plot.py --model llama3_8b --all-batches --mma-candidates "auto"
"""

import argparse
import math
import pathlib

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


def plot_oracle_grid(
    results_by_batch: dict[int, tuple],
    model: str,
    output_suffix: str,
) -> pathlib.Path:
    batch_sizes = list(results_by_batch)
    nrows = len(batch_sizes)
    max_kv_len = max(int(base["kv_len"].max()) for base, _ in results_by_batch.values())

    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(15.62, 9.78),
        sharex="col",
        squeeze=False,
        gridspec_kw={"wspace": 0.16, "hspace": 0.10},
    )

    for row, batch_size in enumerate(batch_sizes):
        base_frame, oracle = results_by_batch[batch_size]
        ax_latency, ax_speedup = axes[row]

        ax_latency.plot(
            base_frame["kv_len"],
            base_frame["default_ms"],
            color="#333333",
            ls="--",
            lw=1.35,
            label="FlashInfer Default",
        )
        ax_latency.plot(
            oracle["kv_len"],
            oracle["ms"],
            color="#D62728",
            lw=1.75,
            label="Oracle",
        )

        ax_speedup.axhline(1.0, color="#333333", ls="--", lw=1.2, label="Default (=1.0)")
        ax_speedup.plot(
            oracle["kv_len"],
            oracle["speedup"],
            color="#1F77B4",
            lw=1.75,
            label="Oracle Speedup",
        )

        if batch_size == 1:
            ymin, ymax = ax_latency.get_ylim()
            ax_latency.set_ylim(ymin, max(ymax, 0.070))

        ax_latency.text(
            0.015,
            0.82,
            f"BS={batch_size}",
            transform=ax_latency.transAxes,
            fontsize=10,
            fontweight="bold",
        )
        for ax in (ax_latency, ax_speedup):
            format_kv_axis(ax, max_kv_len)
            ax.grid(True, which="both", ls=":", alpha=0.40)
            ax.tick_params(axis="x", labelbottom=(row == nrows - 1))

    axes[0, 0].set_title("Latency", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("Oracle Speedup vs FlashInfer Default", fontsize=12, fontweight="bold")
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=True)
    axes[0, 1].legend(loc="upper right", fontsize=8, frameon=True)
    axes[-1, 0].set_xlabel("KV Length")
    axes[-1, 1].set_xlabel("KV Length")

    fig.text(0.015, 0.53, "Latency (ms)", va="center", rotation="vertical", fontsize=11)
    fig.text(0.507, 0.53, "Speedup", va="center", rotation="vertical", fontsize=11)
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
    out = PLOTS_DIR / f"{model}_oracle_latency_speedup_by_batch_{output_suffix}.png"
    fig.tight_layout(rect=(0.035, 0.045, 1, 0.97))
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot oracle latency and speedup as stacked batch panels.")
    parser.add_argument("--csv", type=pathlib.Path, default=CSV_PATH)
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batches", default=None, help="Batch sizes to plot, e.g. '1 2 4 8 16'.")
    parser.add_argument("--all-batches", action="store_true", help="Plot every available batch size.")
    parser.add_argument("--split-modes", default=None, help="Oracle split-k candidates. Default: every non-auto mode.")
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
        split_modes = split_filter or [
            mode for mode in available_split_modes(df, args.model, batch_size) if mode != "auto"
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
        print(f"[batch={batch_size}] oracle NUM_MMA_KV counts:")
        print(oracle["NUM_MMA_KV"].astype(int).value_counts().sort_index().to_string())
        print(
            f"[batch={batch_size}] oracle speedup geo/mean/max: "
            f"{math.exp(oracle['speedup'].map(math.log).mean()):.4f} / "
            f"{oracle['speedup'].mean():.4f} / "
            f"{oracle['speedup'].max():.4f}"
        )

    out = plot_oracle_grid(results_by_batch, args.model, "baseline_before")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
