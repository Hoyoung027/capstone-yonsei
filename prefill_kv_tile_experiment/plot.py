"""
Plot KV tile experiment speedup vs FlashInfer auto baseline.

Input:
    results/data/tile_kv_results.csv

Output:
    results/plots/llama3_8b_speedup_vs_baseline.png
    results/plots/llama3_8b_latency.png

Usage:
    cd /root/capstone-yonsei/kv_tile_experiment
    python plot.py
    python plot.py --model llama3_8b
"""

import argparse
import pathlib
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


ROOT = pathlib.Path(__file__).parent
CSV_PATH = ROOT / "results" / "data" / "tile_kv_results.csv"
PLOTS_DIR = ROOT / "results" / "plots"

MMA_STYLES = {
    1: dict(color="#2e86c1", marker="o"),
    2: dict(color="#27ae60", marker="s"),
    3: dict(color="#8e44ad", marker="v"),
    4: dict(color="#e67e22", marker="^"),
    5: dict(color="#d35400", marker="p"),
    6: dict(color="#16a085", marker="h"),
    7: dict(color="#7f8c8d", marker="x"),
    8: dict(color="#c0392b", marker="D"),
}
    
SEQ_TICKS = [128, 256, 512, 1024, 2048, 4096, 8192]


def format_seq_axis(ax) -> None:
    ax.set_xticks(SEQ_TICKS)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def parse_label(label: str) -> tuple[str, str, int | None]:
    baseline_match = re.fullmatch(r"\[(baseline_before|baseline_after)\]\s+(.+)", label)
    if baseline_match:
        phase, model = baseline_match.groups()
        return model, phase, None

    experiment_match = re.fullmatch(r"\[experiment\]\s+(.+)_num_mma_kv_(\d+)", label)
    if experiment_match:
        model, mma = experiment_match.groups()
        return model, "experiment", int(mma)

    return label, "unknown", None


def load_results(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["seq_len", "CTA_TILE_KV", "NUM_MMA_KV", "ms", "tflops"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    parsed = df["label"].apply(parse_label)
    df["model"] = parsed.apply(lambda x: x[0])
    df["phase"] = parsed.apply(lambda x: x[1])
    df["forced_mma"] = parsed.apply(lambda x: x[2])
    return df


def baseline_series(df: pd.DataFrame, mode: str) -> pd.Series:
    before = df[df["phase"] == "baseline_before"].sort_values("seq_len").set_index("seq_len")["ms"]
    after = df[df["phase"] == "baseline_after"].sort_values("seq_len").set_index("seq_len")["ms"]

    if mode == "before":
        return before
    if mode == "after":
        return after
    if before.empty:
        return after
    if after.empty:
        return before

    common = before.index.intersection(after.index)
    return ((before.loc[common] + after.loc[common]) / 2).rename("baseline_ms")


def plot_speedup(df: pd.DataFrame, model: str, baseline_mode: str) -> pathlib.Path:
    model_df = df[df["model"] == model].copy()
    base = baseline_series(model_df, baseline_mode)
    exps = model_df[model_df["phase"] == "experiment"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(1.0, color="black", lw=1.8, ls="--", label="Baseline (=1.0)")

    summary_lines = []
    for mma, grp in sorted(exps.groupby("forced_mma"), key=lambda item: int(item[0])):
        mma = int(mma)
        g = grp.sort_values("seq_len").set_index("seq_len")
        common = g.index.intersection(base.index)
        speedup = base.loc[common] / g.loc[common, "ms"]
        style = MMA_STYLES.get(mma, {})
        cta_vals = sorted(g.loc[common, "CTA_TILE_KV"].dropna().astype(int).unique().tolist())
        cta_text = cta_vals[0] if len(cta_vals) == 1 else ",".join(map(str, cta_vals))

        ax.plot(
            speedup.index,
            speedup.values,
            lw=2,
            ms=4,
            label=f"NUM_MMA_KV={mma} (CTA_KV={cta_text})",
            **style,
        )

        summary_lines.append(
            f"MMA {mma}: avg {speedup.mean():.3f}x, "
            f"min {speedup.min():.3f}x, max {speedup.max():.3f}x"
        )

    ax.set_title(f"{model} KV Tile Speedup vs FlashInfer Auto Baseline")
    ax.set_xlabel("seq_len")
    ax.set_ylabel("Speedup by latency: baseline_ms / experiment_ms")
    ax.set_xscale("log", base=2)
    format_seq_axis(ax)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(loc="best", fontsize=9)

    if summary_lines:
        ax.text(
            0.02,
            0.02,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9),
        )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_speedup_vs_baseline.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_latency(df: pd.DataFrame, model: str) -> pathlib.Path:
    model_df = df[df["model"] == model].copy()
    fig, ax = plt.subplots(figsize=(10, 6))

    for phase, label, style in [
        ("baseline_before", "baseline_before", dict(color="#111111", ls="--", marker="")),
        ("baseline_after", "baseline_after", dict(color="#666666", ls=":", marker="")),
    ]:
        g = model_df[model_df["phase"] == phase].sort_values("seq_len")
        if not g.empty:
            ax.plot(g["seq_len"], g["ms"], lw=2.2, label=label, **style)

    exps = model_df[model_df["phase"] == "experiment"]
    for mma, grp in sorted(exps.groupby("forced_mma"), key=lambda item: int(item[0])):
        mma = int(mma)
        g = grp.sort_values("seq_len")
        style = MMA_STYLES.get(mma, {})
        ax.plot(g["seq_len"], g["ms"], lw=1.8, ms=3.5, label=f"NUM_MMA_KV={mma}", **style)

    ax.set_title(f"{model} Latency")
    ax.set_xlabel("seq_len")
    ax.set_ylabel("Latency (ms)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    format_seq_axis(ax)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(loc="best", fontsize=9)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_latency.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KV tile speedup vs baseline.")
    parser.add_argument("--csv", type=pathlib.Path, default=CSV_PATH)
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument(
        "--baseline",
        choices=["mean", "before", "after"],
        default="mean",
        help="Baseline reference for speedup. mean uses average of before/after.",
    )
    args = parser.parse_args()

    df = load_results(args.csv)
    if args.model not in set(df["model"]):
        available = ", ".join(sorted(df["model"].unique()))
        raise SystemExit(f"model not found: {args.model}. available: {available}")

    speedup_path = plot_speedup(df, args.model, args.baseline)
    latency_path = plot_latency(df, args.model)
    print(f"saved: {speedup_path}")
    print(f"saved: {latency_path}")


if __name__ == "__main__":
    main()
