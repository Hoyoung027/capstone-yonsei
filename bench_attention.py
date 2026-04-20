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

import argparse
import csv
import re
import pathlib
import torch
import flashinfer
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flashinfer.jit.attention.modules import get_single_prefill_uri

WARMUP  = 100
REPEAT  = 100
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
# Batch Prefill — Uniform
# ──────────────────────────────────────────────

# total_tokens 고정, 분포만 다르게 — uniform / mild / moderate / extreme
# 3개 total 그룹 × 4개 분포 = 12 configs (batch=4 고정)
DEFAULT_RAGGED_CONFIGS = [
    # total = 4096
    [1024, 1024, 1024, 1024],  # uniform
    [512,  768,  1024, 1792],  # mild      (1:1.5:2:3.5)
    [256,  512,  1024, 2304],  # moderate  (1:2:4:9)
    [64,   64,   64,   3904],  # extreme   (1:1:1:61)

    # total = 16384
    [4096, 4096, 4096, 4096],  # uniform
    [2048, 3072, 4096, 7168],  # mild
    [1024, 2048, 4096, 9216],  # moderate
    [256,  256,  256,  15616], # extreme

    # total = 32768
    [8192, 8192, 8192, 8192],  # uniform
    [4096, 6144, 8192, 14336], # mild
    [2048, 4096, 8192, 18432], # moderate
    [512,  512,  512,  31232], # extreme
]


def run_batch_prefill_uniform(seq_lengths, batch_sizes, num_heads=32, head_dim=128, dtype=DTYPE):
    print(f"\n  {'batch':>5}  {'seq_len':>8}  "
          f"{'CTA_Q':>6}  {'CTA_KV':>7}  {'MMA_Q':>6}  {'MMA_KV':>7}  {'WRP_Q':>6}  {'WRP_KV':>7}  "
          f"{'FA2(ms)':>9}  {'FI(ms)':>8}  {'FA2 TFLOPS':>11}  {'FI TFLOPS':>10}  {'FI/FA2':>7}")
    print(f"  {'-'*115}")

    rows = []
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    prev_batch = None
    for batch_size in batch_sizes:
        if prev_batch is not None:
            print(f"  {'·'*115}")
        prev_batch = batch_size

        prev_tile_q = None
        for seq_len in seq_lengths:
            total = batch_size * seq_len
            q = torch.randn(total, num_heads, head_dim, device=DEVICE, dtype=dtype)
            k = torch.randn(total, num_heads, head_dim, device=DEVICE, dtype=dtype)
            v = torch.randn(total, num_heads, head_dim, device=DEVICE, dtype=dtype)

            indptr = torch.arange(0, total + 1, seq_len, dtype=torch.int32, device=DEVICE)
            cu_seqlens = indptr  # same for FA2

            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
            wrapper.plan(
                indptr, indptr,
                num_heads, num_heads, head_dim,
                causal=True,
                q_data_type=dtype,
            )

            tile = get_prefill_tile_params(lambda: wrapper.run(q, k, v))

            ms_fa2 = bench_ms(lambda: flash_attn_varlen_func(
                q, k, v, cu_seqlens, cu_seqlens, seq_len, seq_len, causal=True
            ))
            ms_fi = bench_ms(lambda: wrapper.run(q, k, v))

            flops   = attention_flops(seq_len, seq_len, num_heads, head_dim, causal=True) * batch_size
            tf_fa2  = tflops(flops, ms_fa2)
            tf_fi   = tflops(flops, ms_fi)
            speedup = ms_fa2 / ms_fi

            cta_q  = tile.get("CTA_TILE_Q",  "?")
            cta_kv = tile.get("CTA_TILE_KV", "?")
            mma_q  = tile.get("NUM_MMA_Q",   "?")
            mma_kv = tile.get("NUM_MMA_KV",  "?")
            wrp_q  = tile.get("NUM_WARPS_Q", "?")
            wrp_kv = tile.get("NUM_WARPS_KV","?")

            if prev_tile_q is not None and cta_q != prev_tile_q:
                print(f"  {'·'*115}")
            prev_tile_q = cta_q

            print(f"  {batch_size:>5}  {seq_len:>8}  "
                  f"{str(cta_q):>6}  {str(cta_kv):>7}  {str(mma_q):>6}  {str(mma_kv):>7}  "
                  f"{str(wrp_q):>6}  {str(wrp_kv):>7}  "
                  f"{ms_fa2:>9.4f}  {ms_fi:>8.4f}  {tf_fa2:>11.3f}  {tf_fi:>10.3f}  {speedup:>6.2f}x")

            rows.append({
                "scenario": "batch_prefill_uniform",
                "batch_size": batch_size, "seq_len": seq_len,
                "total_tokens": total,
                "num_heads": num_heads, "head_dim": head_dim,
                "dtype": str(dtype).split(".")[-1],
                **{k: str(v) for k, v in tile.items()},
                "ms_fa2": round(ms_fa2, 4), "ms_fi": round(ms_fi, 4),
                "tflops_fa2": round(tf_fa2, 3), "tflops_fi": round(tf_fi, 3),
                "speedup_fi_vs_fa2": round(speedup, 3),
            })
    return rows


# ──────────────────────────────────────────────
# Batch Prefill — Ragged (혼합 길이)
# ──────────────────────────────────────────────

def run_batch_prefill_ragged(ragged_configs, num_heads=32, head_dim=128, dtype=DTYPE):
    print(f"\n  {'seq_lens (batch)':^30}  {'max':>5}  {'total':>6}  "
          f"{'CTA_Q':>6}  {'CTA_KV':>7}  {'MMA_Q':>6}  {'MMA_KV':>7}  {'WRP_Q':>6}  {'WRP_KV':>7}  "
          f"{'FA2(ms)':>9}  {'FI(ms)':>8}  {'FA2 TFLOPS':>11}  {'FI TFLOPS':>10}  {'FI/FA2':>7}")
    print(f"  {'-'*135}")

    rows = []
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    for lens in ragged_configs:
        import itertools
        cumsum = list(itertools.accumulate(lens, initial=0))
        total = cumsum[-1]
        max_len = max(lens)

        q = torch.randn(total, num_heads, head_dim, device=DEVICE, dtype=dtype)
        k = torch.randn(total, num_heads, head_dim, device=DEVICE, dtype=dtype)
        v = torch.randn(total, num_heads, head_dim, device=DEVICE, dtype=dtype)

        indptr    = torch.tensor(cumsum, dtype=torch.int32, device=DEVICE)
        cu_seqlens = indptr  # FA2도 동일 포맷

        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(
            indptr, indptr,
            num_heads, num_heads, head_dim,
            causal=True,
            q_data_type=dtype,
        )

        tile = get_prefill_tile_params(lambda: wrapper.run(q, k, v))

        ms_fa2 = bench_ms(lambda: flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_len, max_len, causal=True
        ))
        ms_fi = bench_ms(lambda: wrapper.run(q, k, v))

        flops   = sum(attention_flops(l, l, num_heads, head_dim, causal=True) for l in lens)
        tf_fa2  = tflops(flops, ms_fa2)
        tf_fi   = tflops(flops, ms_fi)
        speedup = ms_fa2 / ms_fi

        cta_q  = tile.get("CTA_TILE_Q",  "?")
        cta_kv = tile.get("CTA_TILE_KV", "?")
        mma_q  = tile.get("NUM_MMA_Q",   "?")
        mma_kv = tile.get("NUM_MMA_KV",  "?")
        wrp_q  = tile.get("NUM_WARPS_Q", "?")
        wrp_kv = tile.get("NUM_WARPS_KV","?")

        lens_str = "[" + ",".join(str(l) for l in lens) + "]"
        print(f"  {lens_str:^30}  {max_len:>5}  {total:>6}  "
              f"{str(cta_q):>6}  {str(cta_kv):>7}  {str(mma_q):>6}  {str(mma_kv):>7}  "
              f"{str(wrp_q):>6}  {str(wrp_kv):>7}  "
              f"{ms_fa2:>9.4f}  {ms_fi:>8.4f}  {tf_fa2:>11.3f}  {tf_fi:>10.3f}  {speedup:>6.2f}x")

        rows.append({
            "scenario": "batch_prefill_ragged",
            "seq_lens": lens_str,
            "batch_size": len(lens), "max_seq_len": max_len, "total_tokens": total,
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
    parser = argparse.ArgumentParser(
        description="FlashAttention2 vs FlashInfer 벤치마크"
    )
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    _seq_lengths = (
        [8, 16, 32, 64] +                          # 작은 값
        list(range(128, 8192 + 1, 128)) +           # 128 간격 (128~8192)
        [16384, 32768, 65536, 131072, 262144]        # 2배씩 (8192 이후)
    )
    parser.add_argument("--seq_lengths", type=int, nargs="+",
                        default=_seq_lengths)
    parser.add_argument("--no_prefill", action="store_true")
    parser.add_argument("--no_batch_prefill", action="store_true")
    parser.add_argument("--no_batch_prefill_ragged", action="store_true")
    parser.add_argument("--batch_sizes", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32])
    args = parser.parse_args()

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
    print(SEP)

    all_results = []

    # ── Prefill ──
    if not args.no_prefill:
        print(f"\n{'─'*100}")
        print(f"  PREFILL  (num_heads={args.num_heads}, head_dim={args.head_dim}, causal=True, batch=1)")
        print(f"  CTA_TILE_Q 변화 구간: seq≤16 → 16,  16<seq≤64 → 64,  seq>64 → 128")
        print(f"{'─'*100}")
        all_results += run_prefill(
            seq_lengths=args.seq_lengths,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
        )

    # ── Batch Prefill Uniform ──
    if not args.no_batch_prefill:
        print(f"\n{'─'*100}")
        print(f"  BATCH PREFILL — Uniform  (num_heads={args.num_heads}, head_dim={args.head_dim}, causal=True)")
        print(f"  batch_sizes={args.batch_sizes},  seq_lengths={args.seq_lengths}")
        print(f"{'─'*100}")
        all_results += run_batch_prefill_uniform(
            seq_lengths=args.seq_lengths,
            batch_sizes=args.batch_sizes,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
        )

    # ── Batch Prefill Ragged ──
    if not args.no_batch_prefill_ragged:
        print(f"\n{'─'*100}")
        print(f"  BATCH PREFILL — Ragged  (num_heads={args.num_heads}, head_dim={args.head_dim}, causal=True)")
        print(f"  total_tokens 고정, 분포 변화: uniform / mild / moderate / extreme")
        print(f"{'─'*100}")
        all_results += run_batch_prefill_ragged(
            ragged_configs=DEFAULT_RAGGED_CONFIGS,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
        )

    # ── CSV 저장 ──
    csv_path = RESULTS_DIR / "bench_results.csv"

    # 기존 데이터 로드
    existing = []
    existing_fieldnames = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            existing = list(reader)

    # 이번 실행 시나리오의 기존 행 제거 후 새 결과로 교체
    new_scenarios = {r["scenario"] for r in all_results}
    existing = [r for r in existing if r.get("scenario") not in new_scenarios]

    merged = existing + all_results
    fieldnames = list(dict.fromkeys(
        list(existing_fieldnames) + [k for r in all_results for k in r.keys()]
    ))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(merged)

    print(f"\n{SEP}")
    print(f"  결과 저장: {csv_path}")
    print(SEP)
