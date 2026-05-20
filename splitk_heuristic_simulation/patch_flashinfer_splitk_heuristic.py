#!/usr/bin/env python3
"""Apply or restore a FlashInfer split-k heuristic patch.

The patch is intentionally small and reversible.  It modifies the installed
FlashInfer scheduler so split-auto keeps the original occupancy-oriented
binary-search result, then caps over-splitting with:

    min work per chunk = alpha * CTA_TILE_KV_PROXY tokens
    num_chunks <= floor(beta / batch_size)

This file patches the venv installation used by the capstone experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path


SCHEDULER = Path(
    "/root/capstone-yonsei/venv/lib/python3.10/site-packages/flashinfer/data/include/"
    "flashinfer/attention/scheduler.cuh"
)
BACKUP = SCHEDULER.with_suffix(SCHEDULER.suffix + ".before_splitk_heuristic")

BEGIN = "  // BEGIN capstone split-k heuristic guard\n"
END = "  // END capstone split-k heuristic guard\n"


ORIGINAL = """  } else {
    std::tie(split_kv, kv_chunk_size) = PrefillBinarySearchKVChunkSize(
        enable_cuda_graph, max_batch_size_if_split, packed_qo_len_arr, effective_kv_len_arr,
        cta_tile_q, min_kv_chunk_size);
  }
  // step 3: split qo_indptr and kv_indptr
"""


def patched_block(alpha: int, beta: int, cta_tile_kv_proxy: int) -> str:
    return f"""  }} else {{
    std::tie(split_kv, kv_chunk_size) = PrefillBinarySearchKVChunkSize(
        enable_cuda_graph, max_batch_size_if_split, packed_qo_len_arr, effective_kv_len_arr,
        cta_tile_q, min_kv_chunk_size);
{BEGIN}    constexpr int64_t splitk_alpha = {alpha};
    constexpr int64_t splitk_beta = {beta};
    constexpr int64_t cta_tile_kv_proxy_tokens = {cta_tile_kv_proxy};

    int64_t max_effective_kv_len = 1;
    for (const int64_t& kv_len : effective_kv_len_arr) {{
      max_effective_kv_len = std::max(max_effective_kv_len, kv_len);
    }}

    const int64_t min_work_per_chunk_pages = std::max<int64_t>(
        int64_t(min_kv_chunk_size),
        ceil_div(splitk_alpha * cta_tile_kv_proxy_tokens, int64_t(page_size)));
    const int64_t default_chunks = std::max<int64_t>(1, ceil_div(max_effective_kv_len, kv_chunk_size));
    const int64_t work_cap =
        std::max<int64_t>(1, max_effective_kv_len / min_work_per_chunk_pages);
    const int64_t batch_cap = std::max<int64_t>(1, splitk_beta / int64_t(batch_size));
    const int64_t proposed_chunks =
        std::max<int64_t>(1, std::min(default_chunks, std::min(work_cap, batch_cap)));

    kv_chunk_size = std::max<int64_t>(
        int64_t(min_kv_chunk_size), ceil_div(max_effective_kv_len, proposed_chunks));
    split_kv = enable_cuda_graph || kv_chunk_size < max_effective_kv_len;
{END}  }}
  // step 3: split qo_indptr and kv_indptr
"""


def apply_patch(alpha: int, beta: int, cta_tile_kv_proxy: int) -> None:
    text = SCHEDULER.read_text()
    if BEGIN in text:
        raise SystemExit("scheduler already contains capstone split-k heuristic patch")
    if not BACKUP.exists():
        BACKUP.write_text(text)
    if ORIGINAL not in text:
        raise SystemExit("target block not found; scheduler layout may have changed")
    text = text.replace(ORIGINAL, patched_block(alpha, beta, cta_tile_kv_proxy), 1)
    SCHEDULER.write_text(text)
    print(f"applied patch: {SCHEDULER}")
    print(f"backup: {BACKUP}")
    print(f"alpha={alpha}, beta={beta}, cta_tile_kv_proxy={cta_tile_kv_proxy}")


def restore() -> None:
    if not BACKUP.exists():
        raise SystemExit(f"backup not found: {BACKUP}")
    SCHEDULER.write_text(BACKUP.read_text())
    print(f"restored scheduler from: {BACKUP}")


def status() -> None:
    text = SCHEDULER.read_text()
    if BEGIN in text:
        print("patched")
    else:
        print("not patched")
    print(f"scheduler: {SCHEDULER}")
    print(f"backup_exists: {BACKUP.exists()} ({BACKUP})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch FlashInfer split-k scheduler heuristic.")
    parser.add_argument("command", choices=["apply", "restore", "status"])
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--beta", type=int, default=16)
    parser.add_argument("--cta-tile-kv-proxy", type=int, default=64)
    args = parser.parse_args()

    if args.command == "apply":
        apply_patch(args.alpha, args.beta, args.cta_tile_kv_proxy)
    elif args.command == "restore":
        restore()
    else:
        status()


if __name__ == "__main__":
    main()
