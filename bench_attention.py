"""
FlashAttention2 vs FlashInfer 벤치마크 — Prefill / Decode

실행:
    python bench_attention.py

출력:
    - prefill: seq_len 변화에 따른 실제 타일 설정 + latency / TFLOPS / speedup
    - decode:  kv_len  변화에 따른 실제 타일 설정 + latency / TFLOPS / speedup
    - 타일 설정은 torch profiler로 실제 실행된 CUDA 커널명에서 추출
    - 결과 CSV: results/bench_results.csv
"""

import csv
import re
import pathlib
import torch
import flashinfer
from flash_attn import flash_attn_func
from flashinfer.jit.attention.modules import (
    get_single_decode_uri,
    get_single_prefill_uri,
)

WARMUP  = 50
REPEAT  = 200
DTYPE   = torch.float16
DEVICE  = "cuda"
SEP     = "=" * 100

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# 실제 커널 파라미터 추출 (torch profiler)
# ──────────────────────────────────────────────

def get_prefill_tile_params(fn) -> dict:
    """
    실행된 FlashInfer prefill 커널명에서 타일 설정을 추출한다.

    커널 템플릿: KernelTraits<MASK_MODE,
        CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV,
        NUM_MMA_D_QK, NUM_MMA_D_VO,
        NUM_WARPS_Q, NUM_WARPS_KV, ...>

    CTA_TILE_KV = NUM_MMA_KV * NUM_WARPS_KV * 16
    """
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()

    for e in prof.key_averages():
        m = re.search(
            r'KernelTraits<[^,]+,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,'
            r'\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u',
            e.key,
        )
        if m:
            cta_tile_q   = int(m.group(1))
            num_mma_q    = int(m.group(2))
            num_mma_kv   = int(m.group(3))
            num_warps_q  = int(m.group(6))
            num_warps_kv = int(m.group(7))
            cta_tile_kv  = num_mma_kv * num_warps_kv * 16
            return {
                "CTA_TILE_Q":  cta_tile_q,
                "CTA_TILE_KV": cta_tile_kv,
                "NUM_MMA_Q":   num_mma_q,
                "NUM_MMA_KV":  num_mma_kv,
                "NUM_WARPS_Q": num_warps_q,
                "NUM_WARPS_KV":num_warps_kv,
            }
    return {}


def get_decode_tile_params(fn) -> dict:
    """
    실행된 FlashInfer decode 커널명에서 타일 설정을 추출한다.

    커널 템플릿: SingleDecodeWithKVCacheKernel<
        POS_ENC, NUM_STAGES_SMEM,
        tile_size_per_bdx, vec_size, bdx, bdy, bdz, ...>

    KV_TILE (한 iteration에서 처리하는 KV 토큰 수)
        = tile_size_per_bdx * bdy * bdz
    """
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()

    for e in prof.key_averages():
        m = re.search(
            r'SingleDecodeWithKVCacheKernel<[^,]+,\s*(\d+)u,\s*(\d+)u,'
            r'\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u',
            e.key,
        )
        if m:
            num_stages       = int(m.group(1))
            tile_size_per_bdx= int(m.group(2))
            vec_size         = int(m.group(3))
            bdx              = int(m.group(4))
            bdy              = int(m.group(5))
            bdz              = int(m.group(6))
            kv_tile          = tile_size_per_bdx * bdy * bdz
            return {
                "KV_TILE":          kv_tile,
                "tile_size_per_bdx":tile_size_per_bdx,
                "vec_size":         vec_size,
                "bdx":              bdx,
                "bdy":              bdy,
                "bdz":              bdz,
                "NUM_STAGES_SMEM":  num_stages,
            }
    return {}


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────

def bench_ms(fn, warmup=WARMUP, repeat=REPEAT) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def attention_flops(seq_q, seq_k, num_heads, head_dim, causal) -> float:
    scale = 0.5 if causal else 1.0
    return 4 * num_heads * seq_q * seq_k * head_dim * scale


def tflops(flops, ms) -> float:
    return flops / (ms * 1e-3) / 1e12


# ──────────────────────────────────────────────
# Prefill
# ──────────────────────────────────────────────

def run_prefill(seq_lengths, num_heads=32, head_dim=128, dtype=DTYPE):
    print(f"\n  {'seq_len':>8}  "
          f"{'CTA_Q':>6}  {'CTA_KV':>7}  {'MMA_Q':>6}  {'MMA_KV':>7}  {'WRP_Q':>6}  {'WRP_KV':>7}  "
          f"{'FA2(ms)':>9}  {'FI(ms)':>8}  {'FA2 TFLOPS':>11}  {'FI TFLOPS':>10}  {'FI/FA2':>7}")
    print(f"  {'-'*110}")

    rows = []
    prev_tile_q = None

    for seq_len in seq_lengths:
        q = torch.randn(seq_len, num_heads, head_dim, device=DEVICE, dtype=dtype)
        k = torch.randn(seq_len, num_heads, head_dim, device=DEVICE, dtype=dtype)
        v = torch.randn(seq_len, num_heads, head_dim, device=DEVICE, dtype=dtype)
        q_fa = q.unsqueeze(0); k_fa = k.unsqueeze(0); v_fa = v.unsqueeze(0)

        # 타일 설정 추출 (1회 실행으로)
        tile = get_prefill_tile_params(
            lambda: flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
        )

        # 성능 측정
        ms_fa2 = bench_ms(lambda: flash_attn_func(q_fa, k_fa, v_fa, causal=True))
        ms_fi  = bench_ms(lambda: flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True))

        flops   = attention_flops(seq_len, seq_len, num_heads, head_dim, causal=True)
        tf_fa2  = tflops(flops, ms_fa2)
        tf_fi   = tflops(flops, ms_fi)
        speedup = ms_fa2 / ms_fi

        cta_q   = tile.get("CTA_TILE_Q",  "?")
        cta_kv  = tile.get("CTA_TILE_KV", "?")
        mma_q   = tile.get("NUM_MMA_Q",   "?")
        mma_kv  = tile.get("NUM_MMA_KV",  "?")
        wrp_q   = tile.get("NUM_WARPS_Q", "?")
        wrp_kv  = tile.get("NUM_WARPS_KV","?")

        # 타일 변경 시 구분선
        if prev_tile_q is not None and cta_q != prev_tile_q:
            print(f"  {'·'*110}")
        prev_tile_q = cta_q

        print(f"  {seq_len:>8}  "
              f"{str(cta_q):>6}  {str(cta_kv):>7}  {str(mma_q):>6}  {str(mma_kv):>7}  "
              f"{str(wrp_q):>6}  {str(wrp_kv):>7}  "
              f"{ms_fa2:>9.4f}  {ms_fi:>8.4f}  {tf_fa2:>11.3f}  {tf_fi:>10.3f}  {speedup:>6.2f}x")

        rows.append({
            "scenario": "prefill", "seq_len": seq_len,
            "num_heads": num_heads, "head_dim": head_dim,
            "dtype": str(dtype).split(".")[-1],
            **{k: str(v) for k, v in tile.items()},
            "ms_fa2": round(ms_fa2, 4), "ms_fi": round(ms_fi, 4),
            "tflops_fa2": round(tf_fa2, 3), "tflops_fi": round(tf_fi, 3),
            "speedup_fi_vs_fa2": round(speedup, 3),
        })
    return rows


# ──────────────────────────────────────────────
# Decode
# ──────────────────────────────────────────────

def run_decode(kv_lengths, num_heads=32, head_dim=128, dtype=DTYPE):
    print(f"\n  {'kv_len':>8}  "
          f"{'KV_TILE':>8}  {'tile/bdx':>9}  {'vec_sz':>7}  {'bdx':>4}  {'bdy':>4}  {'bdz':>4}  {'stages':>7}  "
          f"{'FA2(ms)':>9}  {'FI(ms)':>8}  {'FA2 TFLOPS':>11}  {'FI TFLOPS':>10}  {'FI/FA2':>7}")
    print(f"  {'-'*115}")

    rows = []

    for kv_len in kv_lengths:
        q_fi = torch.randn(num_heads, head_dim, device=DEVICE, dtype=dtype)
        k    = torch.randn(kv_len, num_heads, head_dim, device=DEVICE, dtype=dtype)
        v    = torch.randn(kv_len, num_heads, head_dim, device=DEVICE, dtype=dtype)
        q_fa = q_fi.unsqueeze(0).unsqueeze(0)
        k_fa = k.unsqueeze(0); v_fa = v.unsqueeze(0)

        tile = get_decode_tile_params(
            lambda: flashinfer.single_decode_with_kv_cache(q_fi, k, v)
        )

        ms_fa2 = bench_ms(lambda: flash_attn_func(q_fa, k_fa, v_fa, causal=False))
        ms_fi  = bench_ms(lambda: flashinfer.single_decode_with_kv_cache(q_fi, k, v))

        flops   = attention_flops(1, kv_len, num_heads, head_dim, causal=False)
        tf_fa2  = tflops(flops, ms_fa2)
        tf_fi   = tflops(flops, ms_fi)
        speedup = ms_fa2 / ms_fi

        kv_tile = tile.get("KV_TILE",          "?")
        tpb     = tile.get("tile_size_per_bdx", "?")
        vsz     = tile.get("vec_size",          "?")
        bdx     = tile.get("bdx",               "?")
        bdy     = tile.get("bdy",               "?")
        bdz     = tile.get("bdz",               "?")
        stages  = tile.get("NUM_STAGES_SMEM",   "?")

        print(f"  {kv_len:>8}  "
              f"{str(kv_tile):>8}  {str(tpb):>9}  {str(vsz):>7}  "
              f"{str(bdx):>4}  {str(bdy):>4}  {str(bdz):>4}  {str(stages):>7}  "
              f"{ms_fa2:>9.4f}  {ms_fi:>8.4f}  {tf_fa2:>11.3f}  {tf_fi:>10.3f}  {speedup:>6.2f}x")

        rows.append({
            "scenario": "decode", "kv_len": kv_len,
            "num_heads": num_heads, "head_dim": head_dim,
            "dtype": str(dtype).split(".")[-1],
            **{k: str(v) for k, v in tile.items()},
            "ms_fa2": round(ms_fa2, 4), "ms_fi": round(ms_fi, 4),
            "tflops_fa2": round(tf_fa2, 3), "tflops_fi": round(tf_fi, 3),
            "speedup_fi_vs_fa2": round(speedup, 3),
        })
    return rows


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

if __name__ == "__main__":
    props = torch.cuda.get_device_properties(0)
    print(SEP)
    print(f"  FlashAttention2 vs FlashInfer 벤치마크")
    print(f"  GPU      : {props.name}  SM {props.major}.{props.minor}  "
          f"{props.total_memory/1024**3:.0f}GB")
    print(f"  torch {torch.__version__}  "
          f"flash_attn {__import__('flash_attn').__version__}  "
          f"flashinfer {flashinfer.__version__}")
    print(f"\n  [타일 설정은 torch profiler로 실제 실행된 CUDA 커널명에서 추출]")
    print(f"  Prefill 커널: KernelTraits<.., CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, .., NUM_WARPS_Q, NUM_WARPS_KV, ..>")
    print(f"               CTA_TILE_KV = NUM_MMA_KV × NUM_WARPS_KV × 16")
    print(f"  Decode  커널: SingleDecodeWithKVCacheKernel<.., tile_size_per_bdx, vec_size, bdx, bdy, bdz, ..>")
    print(f"               KV_TILE = tile_size_per_bdx × bdy × bdz")
    print(SEP)

    all_results = []

    # ── Prefill ──
    print(f"\n{'─'*100}")
    print("  PREFILL  (num_heads=32, head_dim=128, causal=True, batch=1)")
    print(f"  CTA_TILE_Q 변화 구간: seq≤16 → 16,  16<seq≤64 → 64,  seq>64 → 128")
    print(f"{'─'*100}")
    all_results += run_prefill(
        seq_lengths=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        num_heads=32, head_dim=128,
    )

    # ── Decode ──
    print(f"\n{'─'*100}")
    print("  DECODE  (num_heads=32, head_dim=128, Q seq_len=1)")
    print(f"{'─'*100}")
    all_results += run_decode(
        kv_lengths=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        num_heads=32, head_dim=128,
    )

    # ── CSV 저장 ──
    csv_path = RESULTS_DIR / "bench_results.csv"
    fieldnames = list(dict.fromkeys(k for r in all_results for k in r.keys()))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{SEP}")
    print(f"  결과 저장: {csv_path}")
    print(SEP)
