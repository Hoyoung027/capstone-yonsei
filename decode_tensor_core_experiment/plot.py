"""
Tensor-core decode plot.

// python plot.py --model llama3_8b
// NUM_MMA_KV별 baseline 대비 speedup, latency

// python plot.py --all
// 모든 모델의 NUM_MMA_KV speedup summary

// python plot.py --split --model llama3_8b --mma auto
// split-k 설정별 speedup, latency (NUM_MMA_KV=auto)

// python plot.py --split --model llama3_8b --mma 1
// split-k 설정별 speedup, latency (NUM_MMA_KV=1)

// python plot.py --split --model llama3_8b --mma 2
// split-k 설정별 speedup, latency (NUM_MMA_KV=2)

옵션: --xscale log, --baseline before|after|mean, --csv PATH
"""

import argparse
import math
import pathlib
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


ROOT = pathlib.Path(__file__).parent
CSV_PATH = ROOT / "results" / "data" / "decode_tc_results.csv"
PLOTS_DIR = ROOT / "results" / "plots"
KV_TICKS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

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

SPLIT_STYLES = {
    "auto": dict(color="#111111", marker="o", ls="-"),
    "off": dict(color="#c0392b", marker="x", ls="-"),
    "fixed_16": dict(color="#7f8c8d", marker=".", ls="-"),
    "fixed_32": dict(color="#8e44ad", marker="v", ls="-"),
    "fixed_64": dict(color="#2e86c1", marker="^", ls="-"),
    "fixed_128": dict(color="#16a085", marker="s", ls="-"),
    "fixed_256": dict(color="#27ae60", marker="D", ls="-"),
    "fixed_512": dict(color="#f39c12", marker="p", ls="-"),
    "fixed_1024": dict(color="#d35400", marker="h", ls="-"),
    "fixed_2048": dict(color="#34495e", marker="*", ls="-"),
    "fixed_4096": dict(color="#9b59b6", marker="<", ls="-"),
    "fixed_8192": dict(color="#1abc9c", marker=">", ls="-"),
}


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

    experiment_match = re.fullmatch(r"\[experiment\]\s+(.+)_num_mma_kv_(\d+)", label)
    if experiment_match:
        model, mma = experiment_match.groups()
        return model, "experiment", int(mma)

    return label, "unknown", None


def parse_split_model(model: str) -> tuple[str, str]:
    split_match = re.fullmatch(r"(.+)_split_(auto|off|fixed_\d+)", model)
    if split_match:
        base_model, split_mode = split_match.groups()
        return base_model, split_mode
    return model, "none"


def split_sort_key(split_mode: str) -> tuple[int, int]:
    if split_mode == "auto":
        return (0, 0)
    if split_mode == "off":
        return (1, 0)
    fixed_match = re.fullmatch(r"fixed_(\d+)", split_mode)
    if fixed_match:
        return (2, int(fixed_match.group(1)))
    return (3, 0)


def load_results(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in [
        "kv_len",
        "CTA_TILE_Q",
        "CTA_TILE_KV",
        "NUM_MMA_Q",
        "NUM_MMA_KV",
        "NUM_WARPS_Q",
        "NUM_WARPS_KV",
        "ms",
        "tflops",
        "gb_per_s_est",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    parsed = df["label"].apply(parse_label)
    df["model"] = parsed.apply(lambda x: x[0])
    df["phase"] = parsed.apply(lambda x: x[1])
    df["forced_mma_kv"] = parsed.apply(lambda x: x[2])
    split_parsed = df["model"].apply(parse_split_model)
    df["base_model"] = split_parsed.apply(lambda x: x[0])
    df["split_mode"] = split_parsed.apply(lambda x: x[1])
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


def iter_speedup_series(df: pd.DataFrame, model: str, baseline_mode: str):
    model_df = df[df["model"] == model].copy()
    base = baseline_series(model_df, baseline_mode)
    exps = model_df[model_df["phase"] == "experiment"]
    for mma, grp in sorted(exps.groupby("forced_mma_kv"), key=lambda item: int(item[0])):
        mma = int(mma)
        g = grp.sort_values("kv_len").set_index("kv_len")
        common = g.index.intersection(base.index)
        if common.empty:
            continue
        yield mma, base.loc[common] / g.loc[common, "ms"]


def plot_speedup(df: pd.DataFrame, model: str, baseline_mode: str, xscale: str) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(1.0, color="black", lw=1.8, ls="--", label="Tensor-core auto baseline (=1.0)")

    summary_lines = []
    for mma, speedup in iter_speedup_series(df, model, baseline_mode):
        style = MMA_STYLES.get(mma, {})
        ax.plot(
            speedup.index,
            speedup.values,
            lw=1.35,
            ms=2.8,
            label=f"NUM_MMA_KV={mma}",
            **style,
        )
        summary_lines.append(
            f"mma {mma}: avg {speedup.mean():.3f}x, "
            f"min {speedup.min():.3f}x, max {speedup.max():.3f}x"
        )

    ax.set_title(f"{model} Tensor-Core Decode NUM_MMA_KV Speedup")
    ax.set_xlabel("kv_len")
    ax.set_ylabel("Speedup by latency: baseline_ms / experiment_ms")
    format_kv_axis(ax, xscale)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, frameon=True)

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
    out = PLOTS_DIR / f"{model}_decode_tc_mma_kv_speedup_vs_baseline.png"
    fig.subplots_adjust(right=0.78, bottom=0.25)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_all_model_speedup(df: pd.DataFrame, models: list[str], baseline_mode: str, xscale: str) -> pathlib.Path:
    n_models = len(models)
    ncols = 2 if n_models > 1 else 1
    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 4.8 * nrows), squeeze=False)

    for ax, model in zip(axes.flat, models):
        ax.axhline(1.0, color="black", lw=1.4, ls="--", label="Baseline (=1.0)")
        for mma, speedup in iter_speedup_series(df, model, baseline_mode):
            style = MMA_STYLES.get(mma, {})
            ax.plot(speedup.index, speedup.values, lw=1.1, ms=2.2, label=f"mma={mma}", **style)
        ax.set_title(model)
        ax.set_xlabel("kv_len")
        ax.set_ylabel("Speedup")
        format_kv_axis(ax, xscale)
        ax.grid(True, which="both", ls=":", alpha=0.45)

    for ax in axes.flat[n_models:]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Tensor-Core Decode NUM_MMA_KV Speedup vs Auto Baseline", fontsize=15)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "all_models_decode_tc_mma_kv_speedup_vs_baseline.png"
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

    for mma, grp in sorted(model_df[model_df["phase"] == "experiment"].groupby("forced_mma_kv"),
                           key=lambda item: int(item[0])):
        mma = int(mma)
        g = grp.sort_values("kv_len")
        style = MMA_STYLES.get(mma, {})
        ax.plot(g["kv_len"], g["ms"], lw=1.8, ms=3.0, label=f"NUM_MMA_KV={mma}", **style)

    ax.set_title(f"{model} Tensor-Core Decode Latency")
    ax.set_xlabel("kv_len")
    ax.set_ylabel("Latency (ms)")
    ax.set_yscale("log")
    format_kv_axis(ax, xscale)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(loc="best", fontsize=8)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{model}_decode_tc_mma_kv_latency.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def split_condition_series(model_df: pd.DataFrame, split_mode: str, mma: str,
                           baseline_mode: str) -> pd.Series:
    split_df = model_df[model_df["split_mode"] == split_mode].copy()
    if mma == "auto":
        return baseline_series(split_df, baseline_mode)

    forced_mma = int(mma)
    exp = split_df[split_df["phase"] == "experiment"]
    exp = exp[exp["forced_mma_kv"] == forced_mma]
    return exp.sort_values("kv_len").set_index("kv_len")["ms"]


def available_split_modes(model_df: pd.DataFrame) -> list[str]:
    modes = [m for m in model_df["split_mode"].dropna().unique() if m != "none"]
    return sorted(modes, key=split_sort_key)


def plot_split_latency(df: pd.DataFrame, base_model: str, mma: str, baseline_mode: str,
                       xscale: str) -> pathlib.Path:
    model_df = df[df["base_model"] == base_model].copy()
    modes = available_split_modes(model_df)
    if not modes:
        raise SystemExit(f"split-k results not found for base model: {base_model}")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for mode in modes:
        series = split_condition_series(model_df, mode, mma, baseline_mode)
        if series.empty:
            continue
        style = SPLIT_STYLES.get(mode, {})
        ax.plot(series.index, series.values, lw=1.55, ms=3.0, label=mode, **style)

    ax.set_title(f"{base_model} Tensor-Core Decode Split-k Latency ({mma_label(mma)})")
    ax.set_xlabel("kv_len")
    ax.set_ylabel("Latency (ms)")
    ax.set_yscale("log")
    format_kv_axis(ax, xscale)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=8, frameon=True)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{base_model}_decode_tc_split_k_latency_{mma_file_label(mma)}.png"
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_split_speedup(df: pd.DataFrame, base_model: str, mma: str, baseline_mode: str,
                       xscale: str) -> pathlib.Path:
    model_df = df[df["base_model"] == base_model].copy()
    modes = available_split_modes(model_df)
    if "auto" not in modes:
        raise SystemExit(f"split_auto baseline not found for base model: {base_model}")

    base = split_condition_series(model_df, "auto", mma, baseline_mode)
    if base.empty:
        raise SystemExit(f"split_auto data not found for {base_model}, mma={mma}")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axhline(1.0, color="black", lw=1.8, ls="--", label="split_auto baseline (=1.0)")

    summary_lines = []
    for mode in modes:
        series = split_condition_series(model_df, mode, mma, baseline_mode)
        common = base.index.intersection(series.index)
        if common.empty:
            continue
        speedup = base.loc[common] / series.loc[common]
        style = SPLIT_STYLES.get(mode, {})
        ax.plot(speedup.index, speedup.values, lw=1.35, ms=2.8, label=mode, **style)
        summary_lines.append(
            f"{mode}: avg {speedup.mean():.3f}x, min {speedup.min():.3f}x, max {speedup.max():.3f}x"
        )

    ax.set_title(f"{base_model} Tensor-Core Decode Split-k Speedup ({mma_label(mma)})")
    ax.set_xlabel("kv_len")
    ax.set_ylabel("Speedup by latency: split_auto_ms / split_mode_ms")
    format_kv_axis(ax, xscale)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=8, frameon=True)

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
    out = PLOTS_DIR / f"{base_model}_decode_tc_split_k_speedup_{mma_file_label(mma)}.png"
    fig.subplots_adjust(right=0.76, bottom=0.25)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def mma_label(mma: str) -> str:
    return "NUM_MMA_KV=auto" if mma == "auto" else f"NUM_MMA_KV={mma}"


def mma_file_label(mma: str) -> str:
    return "mma_auto" if mma == "auto" else f"mma_{mma}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tensor-core decode NUM_MMA_KV results.")
    parser.add_argument("--csv", type=pathlib.Path, default=CSV_PATH)
    parser.add_argument("--model", default="llama3_8b")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--split", action="store_true", help="split-k sweep 결과를 split mode별로 plot")
    parser.add_argument("--mma", choices=["auto", "1", "2", "3", "4"], default="auto")
    parser.add_argument("--xscale", choices=["linear", "log"], default="linear")
    parser.add_argument("--baseline", choices=["mean", "before", "after"], default="mean")
    args = parser.parse_args()

    df = load_results(args.csv)
    available_models = sorted(m for m in df["model"].unique() if m and m != "unknown")
    available_base_models = sorted(m for m in df["base_model"].unique() if m and m != "unknown")

    if args.split:
        if args.model not in set(df["base_model"]):
            available = ", ".join(available_base_models)
            raise SystemExit(f"base model not found: {args.model}. available: {available}")
        print(f"saved: {plot_split_speedup(df, args.model, args.mma, args.baseline, args.xscale)}")
        print(f"saved: {plot_split_latency(df, args.model, args.mma, args.baseline, args.xscale)}")
        return

    if args.all:
        overview_path = plot_all_model_speedup(df, available_models, args.baseline, args.xscale)
        print(f"saved: {overview_path}")
        for model in available_models:
            print(f"saved: {plot_speedup(df, model, args.baseline, args.xscale)}")
        return

    if args.model not in set(df["model"]):
        available = ", ".join(available_models)
        raise SystemExit(f"model not found: {args.model}. available: {available}")

    print(f"saved: {plot_speedup(df, args.model, args.baseline, args.xscale)}")
    print(f"saved: {plot_latency(df, args.model, args.xscale)}")


if __name__ == "__main__":
    main()
