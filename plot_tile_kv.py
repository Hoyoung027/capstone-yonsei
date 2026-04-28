"""
tile_kv_results.csv → CTA_TILE_KV 실험 그래프 생성

출력: results/plots/
  - tile_kv_00_latency.png  : seq_len별 레이턴시 (ms) — NUM_MMA_KV별 비교
  - tile_kv_01_tflops.png   : seq_len별 TFLOPS       — NUM_MMA_KV별 비교
  - tile_kv_02_bar.png      : 특정 seq_len에서 NUM_MMA_KV별 막대 비교
"""

import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).parent / "results/data"
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
CSV_PATH    = RESULTS_DIR / "tile_kv_results.csv"

# CTA_TILE_KV 값별 색상
COLORS = {
    16:  "#2e86c1",   # NUM_MMA_KV=1
    32:  "#27ae60",   # NUM_MMA_KV=2 (default 포함)
    64:  "#e67e22",   # NUM_MMA_KV=4
    128: "#c0392b",   # NUM_MMA_KV=8
}

MARKERS = {16: "o", 32: "s", 64: "^", 128: "D"}


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["CTA_TILE_KV"] = pd.to_numeric(df["CTA_TILE_KV"], errors="coerce").astype(int)
    df["NUM_MMA_KV"]  = pd.to_numeric(df["NUM_MMA_KV"],  errors="coerce").astype(int)
    df["seq_len"]     = pd.to_numeric(df["seq_len"],      errors="coerce").astype(int)
    df["ms"]          = pd.to_numeric(df["ms"],           errors="coerce")
    df["tflops"]      = pd.to_numeric(df["tflops"],       errors="coerce")

    # 같은 CTA_TILE_KV끼리는 대표값 하나만 (default 레이블은 제외하고 실험값 우선)
    df["is_default"] = df["label"].str.contains("default")
    df = df.sort_values(["CTA_TILE_KV", "seq_len", "is_default"])
    df = df.drop_duplicates(subset=["CTA_TILE_KV", "seq_len"], keep="first")
    return df.sort_values(["CTA_TILE_KV", "seq_len"]).reset_index(drop=True)


# ── 00: 레이턴시 꺾은선 ──────────────────────────────────────────

def plot_latency(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    for tile_kv, grp in df.groupby("CTA_TILE_KV"):
        mma_kv = grp["NUM_MMA_KV"].iloc[0]
        color  = COLORS.get(tile_kv, "gray")
        marker = MARKERS.get(tile_kv, "o")
        label = f"NUM_MMA_KV={mma_kv}  (CTA_TILE_KV={tile_kv})"
        if tile_kv == 32:
            label += "  [default]"
        ax.plot(
            grp["seq_len"], grp["ms"],
            marker=marker, color=color, lw=2, ms=7,
            label=label,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2g}"))
    ax.set_xlabel("seq_len")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(
        "CTA_TILE_KV 비교 — Latency vs seq_len\n"
        f"RTX 3090 · batch={df['batch_size'].iloc[0]} · "
        f"heads={df['num_qo_heads'].iloc[0]}/{df['num_kv_heads'].iloc[0]} · dim={df['head_dim'].iloc[0]}"
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "tile_kv_00_latency.png", dpi=150)
    plt.close(fig)
    print("  saved: tile_kv_00_latency.png")


# ── 01: TFLOPS 꺾은선 ────────────────────────────────────────────

def plot_tflops(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    for tile_kv, grp in df.groupby("CTA_TILE_KV"):
        mma_kv = grp["NUM_MMA_KV"].iloc[0]
        color  = COLORS.get(tile_kv, "gray")
        marker = MARKERS.get(tile_kv, "o")
        label = f"NUM_MMA_KV={mma_kv}  (CTA_TILE_KV={tile_kv})"
        if tile_kv == 32:
            label += "  [default]"
        ax.plot(
            grp["seq_len"], grp["tflops"],
            marker=marker, color=color, lw=2, ms=7,
            label=label,
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("seq_len")
    ax.set_ylabel("TFLOPS")
    ax.set_title(
        "CTA_TILE_KV 비교 — TFLOPS vs seq_len\n"
        f"RTX 3090 · batch={df['batch_size'].iloc[0]} · "
        f"heads={df['num_qo_heads'].iloc[0]}/{df['num_kv_heads'].iloc[0]} · dim={df['head_dim'].iloc[0]}"
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "tile_kv_01_tflops.png", dpi=150)
    plt.close(fig)
    print("  saved: tile_kv_01_tflops.png")


# ── 02: 막대 비교 (seq_len별 서브플롯) ──────────────────────────

def plot_bar(df):
    seq_lens  = sorted(df["seq_len"].unique())
    tile_kvs  = sorted(df["CTA_TILE_KV"].unique())
    ncols = 3
    nrows = (len(seq_lens) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
    fig.suptitle(
        "CTA_TILE_KV 비교 — seq_len별 Latency / TFLOPS\n"
        f"RTX 3090 · batch={df['batch_size'].iloc[0]}",
        fontsize=12,
    )

    x     = np.arange(len(tile_kvs))
    width = 0.4

    for idx, seq_len in enumerate(seq_lens):
        r, c  = divmod(idx, ncols)
        ax    = axes[r][c]
        sub   = df[df["seq_len"] == seq_len].set_index("CTA_TILE_KV")

        ms_vals    = [sub.loc[t, "ms"]     if t in sub.index else np.nan for t in tile_kvs]
        tflop_vals = [sub.loc[t, "tflops"] if t in sub.index else np.nan for t in tile_kvs]
        colors     = [COLORS.get(t, "gray") for t in tile_kvs]

        ax2 = ax.twinx()
        bars = ax.bar(x - width/2, ms_vals, width, color=colors, alpha=0.85, label="Latency (ms)")
        ax2.bar(x + width/2, tflop_vals, width, color=colors, alpha=0.4, hatch="//", label="TFLOPS")

        # 값 표기
        for xi, (ms, tf) in enumerate(zip(ms_vals, tflop_vals)):
            if not np.isnan(ms):
                ax.text(xi - width/2, ms + ms * 0.02, f"{ms:.2f}", ha="center",
                        va="bottom", fontsize=7.5, fontweight="bold")
            if not np.isnan(tf):
                ax2.text(xi + width/2, tf + tf * 0.02, f"{tf:.1f}", ha="center",
                         va="bottom", fontsize=7.5, color="gray")

        mma_labels = [
            (f"KV={t}\n(MMA={df[df['CTA_TILE_KV']==t]['NUM_MMA_KV'].iloc[0]})\n★default"
             if t == 32 else
             f"KV={t}\n(MMA={df[df['CTA_TILE_KV']==t]['NUM_MMA_KV'].iloc[0]})")
            if not df[df["CTA_TILE_KV"] == t].empty else str(t)
            for t in tile_kvs
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(mma_labels, fontsize=8)
        ax.set_title(f"seq_len = {seq_len:,}", fontsize=10)
        ax.set_ylabel("Latency (ms)", fontsize=8)
        ax2.set_ylabel("TFLOPS", fontsize=8, color="gray")
        ax.grid(True, axis="y", ls=":", alpha=0.4)

    for idx in range(len(seq_lens), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    # 공통 범례
    patches = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS.get(t, "gray"), alpha=0.85,
                       label=f"CTA_TILE_KV={t}")
        for t in tile_kvs
    ]
    fig.legend(handles=patches, loc="lower center", ncol=len(tile_kvs),
               fontsize=9, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(PLOTS_DIR / "tile_kv_02_bar.png", dpi=150)
    plt.close(fig)
    print("  saved: tile_kv_02_bar.png")


# ── 메인 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"reading {CSV_PATH}")
    df = load()
    print(f"  {len(df)} rows, CTA_TILE_KV values: {sorted(df['CTA_TILE_KV'].unique().tolist())}\n")

    plot_latency(df)
    plot_tflops(df)
    plot_bar(df)

    print(f"\ndone → {PLOTS_DIR}/")
