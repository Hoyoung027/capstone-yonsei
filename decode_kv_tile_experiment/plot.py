"""
Plot decode KV tile experiment speedup vs FlashInfer auto baseline.

Input:
    results/data/decode_kv_results.csv

Output:
    results/plots/llama3_8b_decode_speedup_vs_baseline.png
    results/plots/llama3_8b_decode_latency.png
    results/plots/llama3_8b_decode_baseline_drift.png
    results/plots/all_models_decode_speedup_vs_baseline.png

Usage:
    cd /root/capstone-yonsei/decode_kv_tile_experiment
    /root/venv/bin/python plot.py
    /root/venv/bin/python plot.py --all
    /root/venv/bin/python plot.py --all --xscale log
"""

import argparse
import math
import pathlib
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


ROOT = pathlib.Path(__file__).parent
CSV_PATH = ROOT / "results" / "data" / "decode_kv_results.csv"
PLOTS_DIR = ROOT / "results" / "plots"

TILE_STYLES = {
    1: dict(color="#2e86c1", marker="o"),
    2: dict(color="#27ae60", marker="s"),
    3: dict(color="#8e44ad", marker="v"),
    4: dict(color="#e67e22", marker="^"),
    5: dict(color="#d35400", marker="p"),
    6: dict(color="#16a085", marker="h"),
    7: dict(color="#7f8c8d", marker="x"),
    8: dict(color="#c0392b", marker="D"),
}

KV_TICKS = [128, 256, 512, 1024, 2048, 4096, 8192]


def format_kv_axis(ax, xscale: str) -> None:
    if xscale == "log":
        ax.set_xscale("log", base=2)
        ax.set_xticks(KV_TICKS)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        ax.set_xscale("linear")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1024))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(512))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))


def parse_label(label: str) -> tuple[str, str, int | None]:
    baseline_match = re.fullmatch(r"\[(baseline_before|baseline_after)\]\s+(.+)", label)
    if baseline_match:
        phase, model = baseline_match.groups()
        return model, phase, None

    experiment_match = re.fullmatch(r"\[experiment\]\s+(.+)_tile_size_per_bdx_(\d+)", label)
    if experiment_match:
        model, tile = experiment_match.groups()
        return model, "experiment", int(tile)

    return label, "unknown", None


def load_results(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in [
        "kv_len",
        "TILE_SIZE_PER_BDX",
        "KV_TILE_TOKENS",
        "BDX",
        "BDY",
        "BDZ",
        "ms",
        "tflops",
        "gb_per_s_est",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    parsed = df["label"].apply(parse_label)
    df["model"] = parsed.apply(lambda x: x[0])
    df["phase"] = parsed.apply(lambda x: x[1])
    df["forced_tile"] = parsed.apply(lambda x: x[2])
    return df


def baseline_series(df: pd.DataFrame, mode: str) -> pd.Series:
    before = df[df["phase"] == "baseline_before"].sort_values("kv_len").set_index("kv_len")["ms"]
    after = df[df["phase"] == "baseline_after"].sort_values("kv_len").set_index("kv_len")["ms"]

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


def plot_speedup(df: pd.DataFrame, model: str, baseline_mode: str, xscale: str) -> pathlib.Path:
    model_df = df[df["model"] == model].copy()
    base = baseline_series(model_df, baseline_mode)
    exps = model_df[model_df["phase"] == "experiment"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(1.0, color="black", lw=1.8, ls="--", label="Baseline (=1.0)")

    summary_lines = []
    for tile, grp in sorted(exps.groupby("forced_tile"), key=lambda item: int(item[0])):
        tile = int(tile)
        g = grp.sort_values("kv_len").set_index("kv_len")
        common = g.index.intersection(base.index)
        if common.empty:
            continue

        speedup = base.loc[common] / g.loc[common, "ms"]
        style = TILE_STYLES.get(tile, {})
        kv_tiles = sorted(g.loc[common, "KV_TILE_TOKENS"].dropna().astype(int).unique().tolist())
        kv_tile_text = kv_tiles[0] if len(kv_tiles) == 1 else ",".join(map(str, kv_tiles))

        ax.plot(
            speedup.index,
            speedup.values,
            lw=1.35,
            ms=2.4,
            label=f"tile/bdx={tile} (KV_TILE={kv_tile_text})",
            **style,
        )

        summary_lines.append(
            f"tile {tile}: avg {speedup.mean():.3f}x, "
            f"min {speedup.min():.3f}x, max {speedup.max():.3f}x"
        )

    ax.set_title(f"{model} Decode KV Tile Speedup vs FlashInfer Auto Baseline")
    ax.set_xlabel("kv_len")
    ax.set_ylabel("Speedup by latency: baseline_ms / experiment_ms")
    format_kv_axis(ax, xscale)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        fontsize=8,
        frameon=True,
    )

    if summary_lines:
        ax.text(
            1.02,
            0.5,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            fontsize=8,
            va="center",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9),
        )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_decode_speedup_vs_baseline.png"
    fig.subplots_adjust(right=0.78, bottom=0.25)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def iter_speedup_series(df: pd.DataFrame, model: str, baseline_mode: str):
    model_df = df[df["model"] == model].copy()
    base = baseline_series(model_df, baseline_mode)
    exps = model_df[model_df["phase"] == "experiment"]
    for tile, grp in sorted(exps.groupby("forced_tile"), key=lambda item: int(item[0])):
        tile = int(tile)
        g = grp.sort_values("kv_len").set_index("kv_len")
        common = g.index.intersection(base.index)
        if common.empty:
            continue
        kv_tiles = sorted(g.loc[common, "KV_TILE_TOKENS"].dropna().astype(int).unique().tolist())
        kv_tile_text = kv_tiles[0] if len(kv_tiles) == 1 else ",".join(map(str, kv_tiles))
        yield tile, kv_tile_text, base.loc[common] / g.loc[common, "ms"]


def plot_all_model_speedup(
    df: pd.DataFrame, models: list[str], baseline_mode: str, xscale: str
) -> pathlib.Path:
    n_models = len(models)
    ncols = 2 if n_models > 1 else 1
    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 4.8 * nrows), squeeze=False)

    for ax, model in zip(axes.flat, models):
        baseline_line = ax.axhline(1.0, color="black", lw=1.4, ls="--", label="Baseline (=1.0)")
        for tile, kv_tile_text, speedup in iter_speedup_series(df, model, baseline_mode):
            style = TILE_STYLES.get(tile, {})
            ax.plot(
                speedup.index,
                speedup.values,
                lw=1.1,
                ms=2.0,
                label=f"tile/bdx={tile} (KV_TILE={kv_tile_text})",
                **style,
            )

        ax.set_title(model)
        ax.set_xlabel("kv_len")
        ax.set_ylabel("Speedup")
        format_kv_axis(ax, xscale)
        ax.grid(True, which="both", ls=":", alpha=0.45)

    for ax in axes.flat[n_models:]:
        ax.axis("off")

    legend_handles = [baseline_line]
    legend_labels = ["Baseline (=1.0)"]
    for tile in sorted(TILE_STYLES):
        style = TILE_STYLES[tile]
        handle = plt.Line2D(
            [0],
            [0],
            color=style.get("color"),
            marker=style.get("marker"),
            lw=1.1,
            ms=3,
        )
        legend_handles.append(handle)
        legend_labels.append(f"tile/bdx={tile}")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=9,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle("Decode KV Tile Speedup vs FlashInfer Auto Baseline", fontsize=15)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "all_models_decode_speedup_vs_baseline.png"
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_latency(df: pd.DataFrame, model: str, xscale: str) -> pathlib.Path:
    model_df = df[df["model"] == model].copy()
    fig, ax = plt.subplots(figsize=(10, 6))

    for phase, label, style in [
        ("baseline_before", "baseline_before", dict(color="#111111", ls="--", marker="")),
        ("baseline_after", "baseline_after", dict(color="#666666", ls=":", marker="")),
    ]:
        g = model_df[model_df["phase"] == phase].sort_values("kv_len")
        if not g.empty:
            ax.plot(g["kv_len"], g["ms"], lw=2.2, label=label, **style)

    exps = model_df[model_df["phase"] == "experiment"]
    for tile, grp in sorted(exps.groupby("forced_tile"), key=lambda item: int(item[0])):
        tile = int(tile)
        g = grp.sort_values("kv_len")
        style = TILE_STYLES.get(tile, {})
        ax.plot(g["kv_len"], g["ms"], lw=1.8, ms=3.5, label=f"tile/bdx={tile}", **style)

    ax.set_title(f"{model} Decode Latency")
    ax.set_xlabel("kv_len")
    ax.set_ylabel("Latency (ms)")
    ax.set_yscale("log")
    format_kv_axis(ax, xscale)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(loc="best", fontsize=8)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_decode_latency.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_baseline_drift(df: pd.DataFrame, model: str, xscale: str) -> pathlib.Path | None:
    model_df = df[df["model"] == model].copy()
    before = model_df[model_df["phase"] == "baseline_before"].sort_values("kv_len").set_index("kv_len")["ms"]
    after = model_df[model_df["phase"] == "baseline_after"].sort_values("kv_len").set_index("kv_len")["ms"]
    common = before.index.intersection(after.index)
    if common.empty:
        return None

    drift = after.loc[common] / before.loc[common]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(1.0, color="black", lw=1.5, ls="--")
    ax.plot(common, drift.values, lw=2, marker="o", ms=3.5, color="#34495e")
    ax.set_title(f"{model} Decode Baseline Drift")
    ax.set_xlabel("kv_len")
    ax.set_ylabel("baseline_after_ms / baseline_before_ms")
    format_kv_axis(ax, xscale)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.text(
        0.02,
        0.02,
        f"avg {drift.mean():.3f}x, min {drift.min():.3f}x, max {drift.max():.3f}x",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9),
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_decode_baseline_drift.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot decode KV tile speedup vs baseline.")
    parser.add_argument("--csv", type=pathlib.Path, default=CSV_PATH)
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate speedup plots for every model and one combined overview figure.",
    )
    parser.add_argument(
        "--xscale",
        choices=["linear", "log"],
        default="linear",
        help="X-axis scale for kv_len. Default is linear because kv_len is swept uniformly.",
    )
    parser.add_argument(
        "--baseline",
        choices=["mean", "before", "after"],
        default="mean",
        help="Baseline reference for speedup. mean uses average of before/after.",
    )
    args = parser.parse_args()

    df = load_results(args.csv)
    available_models = sorted(m for m in df["model"].unique() if m and m != "unknown")

    if args.all:
        if not available_models:
            raise SystemExit("no models found in CSV")
        overview_path = plot_all_model_speedup(df, available_models, args.baseline, args.xscale)
        print(f"saved: {overview_path}")
        for model in available_models:
            speedup_path = plot_speedup(df, model, args.baseline, args.xscale)
            print(f"saved: {speedup_path}")
        return

    if args.model not in set(df["model"]):
        available = ", ".join(available_models)
        raise SystemExit(f"model not found: {args.model}. available: {available}")

    speedup_path = plot_speedup(df, args.model, args.baseline, args.xscale)
    latency_path = plot_latency(df, args.model, args.xscale)
    drift_path = plot_baseline_drift(df, args.model, args.xscale)

    print(f"saved: {speedup_path}")
    print(f"saved: {latency_path}")
    if drift_path is not None:
        print(f"saved: {drift_path}")


if __name__ == "__main__":
    main()
