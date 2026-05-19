"""
Shared helpers for the decode KV tile experiment.
"""

import re

import torch


WARMUP = 100
REPEAT = 100
TRIALS = 5


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


def get_tensor_core_decode_tile_params(fn) -> dict:
    """
    Extract tensor-core decode tile parameters.

    FlashInfer BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True)
    dispatches through the FA2 batch prefill module, whose kernel name includes
    KernelTraits template fields:
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


def decode_flops(batch_size, kv_len, num_qo_heads, head_dim) -> float:
    # qk dot + softmax-weighted v accumulation, roughly 4 flops per head_dim element.
    return 4 * batch_size * num_qo_heads * kv_len * head_dim


def tflops(flops, ms) -> float:
    return flops / (ms * 1e-3) / 1e12


def estimated_kv_gb(batch_size, kv_len, num_kv_heads, head_dim, dtype_bytes=2) -> float:
    # Read K and V once. This is only a simple bandwidth-oriented proxy.
    return (2 * batch_size * kv_len * num_kv_heads * head_dim * dtype_bytes) / 1e9
