"""
Shared helpers for the decode KV tile experiment.
"""

import re

import torch


WARMUP = 100
REPEAT = 100


def get_decode_tile_params(fn) -> dict:
    """
    Extract BatchDecodeWithPagedKVCacheKernel template parameters.

    Template order in decode.cuh:
      POS_ENCODING_MODE, num_stages_smem, tile_size_per_bdx,
      vec_size, bdx, bdy, bdz, ...
    """
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()

    for event in prof.key_averages():
        match = re.search(
            r"BatchDecodeWithPagedKVCacheKernel<[^,]+,\s*"
            r"(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u",
            event.key,
        )
        if match:
            num_stages_smem = int(match.group(1))
            tile_size_per_bdx = int(match.group(2))
            vec_size = int(match.group(3))
            bdx = int(match.group(4))
            bdy = int(match.group(5))
            bdz = int(match.group(6))
            return {
                "NUM_STAGES_SMEM": num_stages_smem,
                "TILE_SIZE_PER_BDX": tile_size_per_bdx,
                "VEC_SIZE": vec_size,
                "BDX": bdx,
                "BDY": bdy,
                "BDZ": bdz,
                "KV_TILE_TOKENS": tile_size_per_bdx * bdy * bdz,
            }
    return {}


def bench_ms(fn, warmup=WARMUP, repeat=REPEAT) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def decode_flops(batch_size, kv_len, num_qo_heads, head_dim) -> float:
    # qk dot + softmax-weighted v accumulation, roughly 4 flops per head_dim element.
    return 4 * batch_size * num_qo_heads * kv_len * head_dim


def tflops(flops, ms) -> float:
    return flops / (ms * 1e-3) / 1e12


def estimated_kv_gb(batch_size, kv_len, num_kv_heads, head_dim, dtype_bytes=2) -> float:
    # Read K and V once. This is only a simple bandwidth-oriented proxy.
    return (2 * batch_size * kv_len * num_kv_heads * head_dim * dtype_bytes) / 1e9
