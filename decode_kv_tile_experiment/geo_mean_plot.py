"""
Plot geometric mean speedup for each decode KV_TILE size.

Input:
    results/data/decode_kv_results.csv

Output:
    results/plots/<model>_decode_kv_tile_geomean.png
    results/plots/all_models_decode_kv_tile_geomean.png

Usage:
    cd /root/capstone-yonsei/decode_kv_tile_experiment
    /root/capstone-yonsei/venv/bin/python geo_mean_plot.py --model llama3_8b
    /root/capstone-yonsei/venv/bin/python geo_mean_plot.py --model llama3_8b --kv-tiles "8 16 32 64"
    /root/capstone-yonsei/venv/bin/python geo_mean_plot.py --models "llama3_8b llama3_70b"
    /root/capstone-yonsei/venv/bin/python geo_mean_plot.py --all
"""

import argparse
import math
import pathlib
import re

import matplotlib.pyplot as plt
import pandas as pd


ROOT = pathlib.Path(__file__).parent
CSV_PATH = ROOT / "results" / "data" / "decode_kv_results.csv"
PLOTS_DIR = ROOT / "results" / "plots"

TILE_COLORS = {
    8: "#2e86c1",
    16: "#27ae60",
    24: "#8e44ad",
    32: "#e67e22",
    40: "#d35400",
    48: "#16a085",
    56: "#7f8c8d",
    64: "#c0392b",
}


def parse_int_set(text: str | None) -> set[int] | None:
    if not text:
        return None
    return {int(x) for x in text.replace(",", " ").split() if x}


def parse_str_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [x for x in text.replace(",", " ").split() if x]


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
    for col in ["kv_len", "TILE_SIZE_PER_BDX", "KV_TILE_TOKENS", "ms"]:
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


def geometric_mean(values: pd.Series) -> float:
    values = values.dropna()
    values = values[values > 0]
    if values.empty:
        return float("nan")
    return math.exp(values.map(math.log).mean())


def geomean_rows(
    df: pd.DataFrame,
    model: str,
    baseline_mode: str,
    kv_tile_filter: set[int] | None,
) -> pd.DataFrame:
    model_df = df[df["model"] == model].copy()
    base = baseline_series(model_df, baseline_mode)
    exps = model_df[model_df["phase"] == "experiment"]

    rows = []
    for forced_tile, grp in sorted(exps.groupby("forced_tile"), key=lambda item: int(item[0])):
        g = grp.sort_values("kv_len").set_index("kv_len")
        common = g.index.intersection(base.index)
        if common.empty:
            continue

        kv_tiles = sorted(g.loc[common, "KV_TILE_TOKENS"].dropna().astype(int).unique().tolist())
        if not kv_tiles:
            continue
        if kv_tile_filter is not None and not any(kv in kv_tile_filter for kv in kv_tiles):
            continue

        speedup = base.loc[common] / g.loc[common, "ms"]
        kv_tile = kv_tiles[0] if len(kv_tiles) == 1 else None
        rows.append({
            "forced_tile": int(forced_tile),
            "kv_tile": kv_tile,
            "n": int(speedup.count()),
            "geo_mean_speedup": geometric_mean(speedup),
            "arith_mean_speedup": float(speedup.mean()),
            "min_speedup": float(speedup.min()),
            "max_speedup": float(speedup.max()),
        })

    return pd.DataFrame(rows).sort_values("kv_tile")


def plot_geomean(summary: pd.DataFrame, model: str, baseline_mode: str) -> pathlib.Path:
    if summary.empty:
        raise SystemExit("no rows to plot")

    x = summary["kv_tile"].astype(int).tolist()
    y = summary["geo_mean_speedup"].tolist()
    colors = [TILE_COLORS.get(kv, "#4c78a8") for kv in x]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    bars = ax.bar([str(kv) for kv in x], y, color=colors, edgecolor="#333333", linewidth=0.6)
    ax.axhline(1.0, color="black", lw=1.5, ls="--", label="Baseline (=1.0)")

    for bar, value in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(f"{model} Decode KV_TILE Geomean Speedup")
    ax.set_xlabel("KV_TILE (tokens)")
    ax.set_ylabel("Geomean Latency Speedup")
    ax.grid(True, axis="y", ls=":", alpha=0.45)
    ax.legend(loc="best", fontsize=9)
    fig.text(
        0.5,
        0.015,
        f"Speedup = baseline_ms / experiment_ms over kv_len points; baseline={baseline_mode}.",
        ha="center",
        fontsize=8,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_decode_kv_tile_geomean.png"
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_multi_model_geomean(
    summaries: dict[str, pd.DataFrame], baseline_mode: str, suffix: str = "all_models"
) -> pathlib.Path:
    summaries = {model: summary for model, summary in summaries.items() if not summary.empty}
    if not summaries:
        raise SystemExit("no rows to plot")

    models = list(summaries)
    all_kv_tiles = sorted({int(kv) for summary in summaries.values() for kv in summary["kv_tile"]})
    n_models = len(models)
    width = 0.78 / max(n_models, 1)
    centers = list(range(len(all_kv_tiles)))

    fig, ax = plt.subplots(figsize=(max(8.8, 1.1 * len(all_kv_tiles) + 1.8 * n_models), 5.4))
    ax.axhline(1.0, color="black", lw=1.5, ls="--", label="Baseline (=1.0)")

    for model_idx, model in enumerate(models):
        summary = summaries[model].set_index("kv_tile")
        offset = (model_idx - (n_models - 1) / 2) * width
        xs = [center + offset for center in centers]
        ys = [summary.loc[kv, "geo_mean_speedup"] if kv in summary.index else float("nan") for kv in all_kv_tiles]
        bars = ax.bar(xs, ys, width=width, label=model, edgecolor="#333333", linewidth=0.5)
        for bar, value in zip(bars, ys):
            if pd.isna(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}x",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90 if n_models > 2 else 0,
            )

    ax.set_title("Decode KV_TILE Geomean Speedup by Model")
    ax.set_xlabel("KV_TILE (tokens)")
    ax.set_ylabel("Geomean Latency Speedup")
    ax.set_xticks(centers)
    ax.set_xticklabels([str(kv) for kv in all_kv_tiles])
    ax.grid(True, axis="y", ls=":", alpha=0.45)
    ax.legend(loc="best", fontsize=9)
    fig.text(
        0.5,
        0.015,
        f"Speedup = baseline_ms / experiment_ms over kv_len points; baseline={baseline_mode}.",
        ha="center",
        fontsize=8,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{suffix}_decode_kv_tile_geomean.png"
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot decode KV_TILE geometric mean speedup.")
    parser.add_argument("--csv", type=pathlib.Path, default=CSV_PATH)
    parser.add_argument("--model", default="llama3_8b", help="Single model to plot.")
    parser.add_argument("--models", default=None, help="Space/comma-separated model list for one combined plot.")
    parser.add_argument("--all", action="store_true", help="Plot all models found in the CSV in one combined plot.")
    parser.add_argument(
        "--baseline",
        choices=["mean", "before", "after"],
        default="mean",
        help="Baseline reference for speedup. mean uses average of before/after.",
    )
    parser.add_argument(
        "--kv-tiles",
        default=None,
        help="Only plot selected KV_TILE values, e.g. '8 16 32 64' or '8,16,32,64'.",
    )
    args = parser.parse_args()

    df = load_results(args.csv)
    available_models = sorted(m for m in df["model"].unique() if m and m != "unknown")
    kv_tile_filter = parse_int_set(args.kv_tiles)

    if args.all or args.models:
        models = available_models if args.all else parse_str_list(args.models)
        missing = [model for model in models if model not in set(df["model"])]
        if missing:
            available = ", ".join(available_models)
            raise SystemExit(f"model not found: {', '.join(missing)}. available: {available}")

        summaries = {
            model: geomean_rows(df, model, args.baseline, kv_tile_filter)
            for model in models
        }
        suffix = "all_models" if args.all else "_".join(models)
        out = plot_multi_model_geomean(summaries, args.baseline, suffix=suffix)
        for model, summary in summaries.items():
            print(f"\n[{model}]")
            print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
        print(f"saved: {out}")
        return

    if args.model not in set(df["model"]):
        available = ", ".join(available_models)
        raise SystemExit(f"model not found: {args.model}. available: {available}")

    summary = geomean_rows(df, args.model, args.baseline, kv_tile_filter)
    out = plot_geomean(summary, args.model, args.baseline)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
