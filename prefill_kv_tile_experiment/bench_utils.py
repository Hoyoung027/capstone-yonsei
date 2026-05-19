"""
Shared benchmarking utilities.

This module intentionally contains only small, reusable helpers used by
the tile-KV experiment and the broader attention benchmark.
"""

import re

import torch


WARMUP = 100
REPEAT = 100
TRIALS = 5


def get_prefill_tile_params(fn) -> dict:
    """
    Extract FlashInfer prefill tile parameters from the executed CUDA kernel name.

    KernelTraits template fields include:
      CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV,
      NUM_MMA_D_QK, NUM_MMA_D_VO,
      NUM_WARPS_Q, NUM_WARPS_KV, ...

    CTA_TILE_KV = NUM_MMA_KV * NUM_WARPS_KV * 16
    """
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()

    for event in prof.key_averages():
        match = re.search(
            r"KernelTraits<[^,]+,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,"
            r"\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u",
            event.key,
        )
        if match:
            cta_tile_q = int(match.group(1))
            num_mma_q = int(match.group(2))
            num_mma_kv = int(match.group(3))
            num_warps_q = int(match.group(6))
            num_warps_kv = int(match.group(7))
            cta_tile_kv = num_mma_kv * num_warps_kv * 16
            return {
                "CTA_TILE_Q": cta_tile_q,
                "CTA_TILE_KV": cta_tile_kv,
                "NUM_MMA_Q": num_mma_q,
                "NUM_MMA_KV": num_mma_kv,
                "NUM_WARPS_Q": num_warps_q,
                "NUM_WARPS_KV": num_warps_kv,
            }
    return {}


def bench_ms(fn, warmup=WARMUP, repeat=REPEAT, trials=TRIALS) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    trial_ms = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        trial_ms.append(start.elapsed_time(end) / repeat)

    return float(torch.tensor(trial_ms).median().item())


def attention_flops(seq_q, seq_k, num_heads, head_dim, causal) -> float:
    scale = 0.5 if causal else 1.0
    return 4 * num_heads * seq_q * seq_k * head_dim * scale


def tflops(flops, ms) -> float:
    return flops / (ms * 1e-3) / 1e12
