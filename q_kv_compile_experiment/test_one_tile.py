"""
단일 (Q_tile, KV_tile) 조합 테스트 — subprocess로 호출됨

사용법:
    python test_one_tile.py <q_tile> <kv_tile> [seq_len]

stdout에 JSON 한 줄 출력:
    {"status": "ok", "latency_ms": 1.23, "output_shape": [1024, 8, 128]}
    {"status": "cuda_error", "error": "..."}
    {"status": "compile_error", "error": "..."}
"""

import json
import sys
import traceback

import torch

NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM     = 128
PAGE_SIZE    = 16
BATCH_SIZE   = 4
NUM_WARMUP   = 3
NUM_ITERS    = 20
DTYPE        = torch.float16
DEVICE       = "cuda"


def make_paged_kv(seq_lens):
    pages = [(s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens]
    total = sum(pages)
    kv_cache = torch.randn(total, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                           dtype=DTYPE, device=DEVICE)
    indptr = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=DEVICE)
    indices, last_len = [], []
    offset = 0
    for i, (s, n) in enumerate(zip(seq_lens, pages)):
        indptr[i + 1] = indptr[i] + n
        indices.extend(range(offset, offset + n))
        last_len.append(s - (n - 1) * PAGE_SIZE)
        offset += n
    return (kv_cache,
            indptr,
            torch.tensor(indices, dtype=torch.int32, device=DEVICE),
            torch.tensor(last_len, dtype=torch.int32, device=DEVICE))


def run(seq_len: int):
    import flashinfer

    seq_lens   = [seq_len] * BATCH_SIZE
    qo_indptr  = torch.arange(0, (BATCH_SIZE + 1) * seq_len, seq_len,
                               dtype=torch.int32, device=DEVICE)
    q          = torch.randn(BATCH_SIZE * seq_len, NUM_QO_HEADS, HEAD_DIM,
                             dtype=DTYPE, device=DEVICE)
    kv_cache, kv_indptr, kv_indices, kv_last = make_paged_kv(seq_lens)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper   = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_last,
                 NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE)

    for _ in range(NUM_WARMUP):
        out = wrapper.run(q, kv_cache)
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(NUM_ITERS):
        out = wrapper.run(q, kv_cache)
    t1.record()
    torch.cuda.synchronize()

    ms = t0.elapsed_time(t1) / NUM_ITERS
    return ms, list(out.shape)


def main():
    q_tile  = int(sys.argv[1])
    kv_tile = int(sys.argv[2])
    seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 1024

    try:
        ms, shape = run(seq_len)
        print(json.dumps({"status": "ok", "latency_ms": round(ms, 4),
                          "output_shape": shape}))
    except RuntimeError as e:
        msg = str(e)
        kind = "cuda_error" if "CUDA" in msg or "cuda" in msg else "runtime_error"
        print(json.dumps({"status": kind, "error": msg[:400]}))
    except Exception as e:
        print(json.dumps({"status": "compile_error",
                          "error": traceback.format_exc()[-400:]}))


if __name__ == "__main__":
    main()
