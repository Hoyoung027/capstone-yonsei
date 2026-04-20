"""
bench_results.csv → 그래프 생성

출력: results/plots/
  - 00_prefill_latency.png         : Single Prefill — FA2 vs FI latency + 타일/speedup 주석
  - 01_batch_uniform_latency.png   : Batch Uniform — batch_size별 서브플롯 + 타일/speedup 주석
  - 02_batch_ragged_latency.png    : Batch Ragged — config별 latency + 타일/speedup 주석
"""

import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
CSV_PATH    = RESULTS_DIR / "bench_results.csv"

# A100 80GB float16 이론 peak (312 TFLOPS)
PEAK_TFLOPS = 312.0

TILE_COLORS = {16: "#d4e6f1", 64: "#d5f5e3", 128: "#fdebd0"}
FA2_COLOR   = "#2e86c1"
FI_COLOR    = "#e67e22"


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    for col in ["ms_fa2", "ms_fi", "tflops_fa2", "tflops_fi", "speedup_fi_vs_fa2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df



# ── 00: Single Prefill Latency ───────────────────────────────

def plot_prefill_latency(df):
    d = df[df["scenario"] == "prefill"].sort_values("seq_len").copy()
    if d.empty:
        return

    d["CTA_TILE_Q"]  = pd.to_numeric(d["CTA_TILE_Q"],  errors="coerce")
    d["CTA_TILE_KV"] = pd.to_numeric(d["CTA_TILE_KV"], errors="coerce")

    xs = list(range(len(d)))
    seq_labels = d["seq_len"].astype(int).tolist()

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 7),
        gridspec_kw={"height_ratios": [3, 1.2]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08)

    # ── latency 꺾은선 ──
    ax_top.plot(xs, d["ms_fa2"], "o-", color=FA2_COLOR,
                label="FlashAttention2", lw=2, ms=5)
    ax_top.plot(xs, d["ms_fi"],  "s-", color=FI_COLOR,
                label="FlashInfer",     lw=2, ms=5)

    # ── FI 각 포인트 위에 Q/KV 타일 + speedup 주석 ──
    for i, (_, row) in enumerate(d.iterrows()):
        tq  = int(row["CTA_TILE_Q"])  if not np.isnan(row["CTA_TILE_Q"])  else "?"
        tkv = int(row["CTA_TILE_KV"]) if not np.isnan(row["CTA_TILE_KV"]) else "?"
        spd = row["speedup_fi_vs_fa2"]
        ax_top.annotate(
            f"Q={tq}\nKV={tkv}\n{spd:.2f}×",
            xy=(i, row["ms_fi"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=6.5, color=FI_COLOR,
            arrowprops=dict(arrowstyle="-", color=FI_COLOR, lw=0.6),
        )

    ax_top.set_ylabel("Latency (ms)")
    ax_top.set_title(
        "Single Prefill — FA2 vs FlashInfer Latency\n"
        "(annotations on FI points: CTA_TILE_Q / CTA_TILE_KV / speedup)"
    )
    ax_top.legend(fontsize=8)
    ax_top.grid(True, ls=":", alpha=0.5)

    # ── 하단: speedup 막대 ──
    ax_bot.bar(xs, d["speedup_fi_vs_fa2"],
               color="#8e44ad", alpha=0.7, label="Speedup (FI/FA2)")
    ax_bot.axhline(1.0, color="black", ls="--", lw=1)
    ax_bot.set_ylabel("Speedup", fontsize=9)
    ax_bot.set_xlabel("seq_len")
    ax_bot.set_xticks(xs)
    ax_bot.set_xticklabels(seq_labels, rotation=45, ha="right")
    ax_bot.grid(True, axis="y", ls=":", alpha=0.5)
    ax_bot.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "00_prefill_latency.png", dpi=150)
    plt.close(fig)
    print("  saved: 00_prefill_latency.png")


# ── 01: Batch Prefill Uniform Latency ───────────────────────

def _annotate_tile_speedup(ax, x, y, tq, tkv, spd):
    label = f"Q={tq}\nKV={tkv}\n{spd:.2f}×"
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center", va="bottom",
        fontsize=5.5,
        color=FI_COLOR,
        arrowprops=dict(arrowstyle="-", color=FI_COLOR, lw=0.5),
    )


def plot_batch_uniform_latency(df):
    d = df[df["scenario"] == "batch_prefill_uniform"].copy()
    if d.empty:
        return

    for col in ["batch_size", "seq_len", "CTA_TILE_Q", "CTA_TILE_KV"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    batch_sizes = sorted(d["batch_size"].dropna().unique().astype(int))
    ncols = 3
    nrows = (len(batch_sizes) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), squeeze=False)

    for idx, bs in enumerate(batch_sizes):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sub = d[d["batch_size"] == bs].sort_values("seq_len").reset_index(drop=True)
        xs  = list(range(len(sub)))

        ax.plot(xs, sub["speedup_fi_vs_fa2"], "D-",
                color="#8e44ad", lw=2, ms=5, label="Speedup (FI/FA2)")
        ax.axhline(1.0, color="black", ls="--", lw=1)

        for i, (_, row) in enumerate(sub.iterrows()):
            tq  = int(row["CTA_TILE_Q"])  if not np.isnan(row["CTA_TILE_Q"])  else "?"
            tkv = int(row["CTA_TILE_KV"]) if not np.isnan(row["CTA_TILE_KV"]) else "?"
            ax.annotate(
                f"Q={tq}\nKV={tkv}",
                xy=(i, row["speedup_fi_vs_fa2"]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=5.5, color="#8e44ad",
                arrowprops=dict(arrowstyle="-", color="#8e44ad", lw=0.5),
            )

        ax.set_xticks(xs)
        ax.set_xticklabels(sub["seq_len"].astype(int).tolist(), rotation=45, ha="right")
        ax.set_title(f"batch_size = {bs}")
        ax.set_xlabel("seq_len")
        ax.set_ylabel("Speedup (FI / FA2)")
        ax.legend(fontsize=7)
        ax.grid(True, ls=":", alpha=0.5)

    for idx in range(len(batch_sizes), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        "Batch Prefill Uniform — FlashInfer Speedup vs FA2\n"
        "(annotations: CTA_TILE_Q / CTA_TILE_KV)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_batch_uniform_latency.png", dpi=150)
    plt.close(fig)
    print("  saved: 01_batch_uniform_latency.png")


# ── 02: Batch Prefill Ragged Latency ────────────────────────

def plot_batch_ragged_latency(df):
    d = df[df["scenario"] == "batch_prefill_ragged"].copy()
    if d.empty:
        return

    for col in ["total_tokens", "CTA_TILE_Q", "CTA_TILE_KV"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.sort_values(["total_tokens", "seq_lens"]).reset_index(drop=True)

    totals = sorted(d["total_tokens"].unique())
    n_dist = 4  # uniform / mild / moderate / extreme
    dist_labels = ["uniform", "mild", "moderate", "extreme"]
    group_colors = {t: c for t, c in zip(totals, ["#8e44ad", "#2e86c1", "#27ae60"])}

    # 그룹 사이 간격을 위해 x 위치를 직접 계산
    gap = 0.8
    xs = []
    offset = 0
    for i, t in enumerate(totals):
        for j in range(n_dist):
            xs.append(offset + j)
        offset += n_dist + gap

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, (_, row) in enumerate(d.iterrows()):
        t   = row["total_tokens"]
        bar = ax.bar(xs[i], row["speedup_fi_vs_fa2"], color=group_colors[t], alpha=0.85, width=0.7)
        tq  = int(row["CTA_TILE_Q"])  if not np.isnan(row["CTA_TILE_Q"])  else "?"
        tkv = int(row["CTA_TILE_KV"]) if not np.isnan(row["CTA_TILE_KV"]) else "?"
        ax.text(
            xs[i], row["speedup_fi_vs_fa2"] + 0.03,
            f"{row['speedup_fi_vs_fa2']:.2f}×\nQ={tq}\nKV={tkv}",
            ha="center", va="bottom", fontsize=7,
            color=group_colors[t],
        )

    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_ylim(0, 1.5)

    # 그룹 구분선 + total_tokens 레이블
    offset = 0
    for t in totals:
        center = offset + (n_dist - 1) / 2
        ax.text(center, ax.get_ylim()[0] - 0.25, f"total={int(t)}",
                ha="center", va="top", fontsize=8, color=group_colors[t], fontweight="bold")
        if offset > 0:
            ax.axvline(offset - gap / 2, color="gray", ls=":", lw=1, alpha=0.5)
        offset += n_dist + gap

    ax.set_xticks(xs)
    ax.set_xticklabels(
        dist_labels * len(totals),
        rotation=30, ha="right", fontsize=7,
    )
    ax.set_xlabel("distribution  (batch_size=4, fixed)")
    ax.set_ylabel("Speedup (FI / FA2)")
    ax.set_title(
        "Batch Prefill Ragged — FlashInfer Speedup vs FA2\n"
        "(fixed total_tokens per group, varying distribution)"
    )

    legend_patches = [
        mpatches.Patch(color=c, alpha=0.85, label=f"total_tokens={int(t)}")
        for t, c in group_colors.items()
    ]
    ax.legend(handles=legend_patches, fontsize=8)
    ax.grid(True, axis="y", ls=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_batch_ragged_latency.png", dpi=150)
    plt.close(fig)
    print("  saved: 02_batch_ragged_latency.png")


# ── 메인 ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"reading {CSV_PATH}")
    df = load()
    print(f"  {len(df)} rows, scenarios: {df['scenario'].unique().tolist()}\n")

    plot_prefill_latency(df)
    plot_batch_uniform_latency(df)
    plot_batch_ragged_latency(df)

    print(f"\ndone → {PLOTS_DIR}/")
