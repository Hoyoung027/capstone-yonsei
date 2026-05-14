"""
Tensor-core decode KV tile experiment -- BatchDecodeWithPagedKVCacheWrapper.

FlashInfer batch decode with use_tensor_cores=True internally uses the FA2 batch
prefill module. This benchmark measures decode latency while patching the FA2
KernelTraits NUM_MMA_KV value in prefill.cuh.

Results are written to dtype-specific CSV files under results/data/.
"""

import argparse
import csv
import math
import os
import pathlib

import torch
import flashinfer

from bench_utils import (
    bench_ms,
    decode_flops,
    estimated_kv_gb,
    get_tensor_core_decode_tile_params,
    tflops,
)


RESULTS_DIR = pathlib.Path(__file__).parent / "results"
DATA_DIR = RESULTS_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_KV_LENS = list(range(128, 8193, 128))
DEVICE = "cuda"
SEP = "=" * 120


def parse_dtype(dtype_name):
    if dtype_name in {"float16", "fp16", "half"}:
        return torch.float16
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def dtype_label(dtype):
    if dtype is torch.float16:
        return "fp16"
    if dtype is torch.bfloat16:
        return "bf16"
    return str(dtype)


def dtype_csv_suffix(dtype):
    label = dtype_label(dtype)
    if label == "fp16":
        return "fp16"
    if label == "bf16":
        return "bf16"
    return label.replace(".", "_").replace(":", "_")


def make_paged_kv(seq_lens, num_kv_heads, head_dim, page_size, dtype):
    pages_per_seq = [(s + page_size - 1) // page_size for s in seq_lens]
    total_pages = sum(pages_per_seq)

    paged_kv_cache = torch.randn(
        total_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=DEVICE,
    )

    indptr = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=DEVICE)
    indices_list = []
    last_page_len = []
    page_offset = 0
    for i, (seq_len, n_pages) in enumerate(zip(seq_lens, pages_per_seq)):
        indptr[i + 1] = indptr[i] + n_pages
        indices_list.extend(range(page_offset, page_offset + n_pages))
        last_page_len.append(seq_len - (n_pages - 1) * page_size)
        page_offset += n_pages

    return (
        paged_kv_cache,
        indptr,
        torch.tensor(indices_list, dtype=torch.int32, device=DEVICE),
        torch.tensor(last_page_len, dtype=torch.int32, device=DEVICE),
    )


def make_wrapper(kv_indptr, kv_indices, kv_last, num_qo_heads, num_kv_heads, head_dim,
                 page_size, backend, fixed_split_size=None, disable_split_kv=False):
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_tensor_cores=True,
        backend=backend,
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
    )
    return wrapper


def reconstruct_dense_kv(kv_cache, kv_indptr, kv_indices, kv_last, seq_lens, page_size):
    batch_size = len(seq_lens)
    max_seq_len = max(seq_lens)
    num_kv_heads = kv_cache.shape[3]
    head_dim = kv_cache.shape[4]
    k = torch.empty(
        batch_size,
        max_seq_len,
        num_kv_heads,
        head_dim,
        dtype=kv_cache.dtype,
        device=kv_cache.device,
    )
    v = torch.empty_like(k)

    indptr_cpu = kv_indptr.cpu()
    indices_cpu = kv_indices.cpu()
    last_cpu = kv_last.cpu()
    for b, seq_len in enumerate(seq_lens):
        n_pages = indptr_cpu[b + 1].item() - indptr_cpu[b].item()
        token_off = 0
        for p in range(n_pages):
            pid = indices_cpu[indptr_cpu[b].item() + p].item()
            n_tok = page_size if p < n_pages - 1 else last_cpu[b].item()
            k[b, token_off:token_off + n_tok] = kv_cache[pid, 0, :n_tok]
            v[b, token_off:token_off + n_tok] = kv_cache[pid, 1, :n_tok]
            token_off += n_tok
    return k, v


@torch.no_grad()
def decode_reference(q, kv_cache, kv_indptr, kv_indices, kv_last, seq_lens, page_size):
    k, v = reconstruct_dense_kv(kv_cache, kv_indptr, kv_indices, kv_last, seq_lens, page_size)
    group_size = q.shape[1] // k.shape[2]

    qf = q.float()
    out = torch.empty_like(qf)
    scale = 1.0 / math.sqrt(q.shape[-1])
    for kv_head in range(k.shape[2]):
        head_begin = kv_head * group_size
        head_end = head_begin + group_size
        q_group = qf[:, head_begin:head_end]
        k_head = k[:, :, kv_head].float()
        v_head = v[:, :, kv_head].float()
        scores = torch.einsum("bgd,bsd->bgs", q_group, k_head) * scale
        probs = torch.softmax(scores, dim=-1)
        out[:, head_begin:head_end] = torch.einsum("bgs,bsd->bgd", probs, v_head)
    return out.to(q.dtype)


def correctness_check(q, kv_cache, kv_indptr, kv_indices, kv_last, seq_lens, page_size, out_fi):
    out_ref = decode_reference(q, kv_cache, kv_indptr, kv_indices, kv_last, seq_lens, page_size)
    max_err = (out_fi - out_ref).abs().max().item()
    mean_err = (out_fi - out_ref).abs().mean().item()
    status = "OK" if max_err < 1e-2 else "FAIL"
    return f"max_err={max_err:.5f}  mean_err={mean_err:.5f}  [{status}]"


def parse_int_list(text):
    return [int(x) for x in text.replace(",", " ").split() if x]


def parse_optional_int(text):
    if text is None or str(text).lower() in {"", "none", "-1"}:
        return None
    return int(text)


def summarize_int_list(values):
    if not values:
        return "empty"
    if len(values) == 1:
        return str(values[0])
    step = values[1] - values[0]
    is_regular = all(values[i] - values[i - 1] == step for i in range(1, len(values)))
    if is_regular:
        return f"{values[0]}..{values[-1]} step {step} ({len(values)} values)"
    preview = " ".join(map(str, values[:8]))
    suffix = " ..." if len(values) > 8 else ""
    return f"{preview}{suffix} ({len(values)} values)"


def run(label, num_qo_heads, num_kv_heads, head_dim, batch_size, page_size, kv_lens,
        backend, dtype, fixed_split_size=None, disable_split_kv=False, skip_correctness=False):
    print(SEP)
    print(f"  label={label}")
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {props.name}  SM{props.major}.{props.minor}")
    print(f"  batch={batch_size}  heads={num_qo_heads}/{num_kv_heads}  "
          f"dim={head_dim}  page={page_size}  dtype={dtype_label(dtype)}")
    print(f"  wrapper: BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True, backend={backend})")
    print(f"  fixed_split_size={fixed_split_size}  disable_split_kv={disable_split_kv}")
    print(f"  kv_len: {summarize_int_list(kv_lens)}")
    print(SEP)

    print(f"\n  {'kv_len':>8}  {'CTA_Q':>6}  {'CTA_KV':>7}  {'MMA_KV':>7}  "
          f"{'WRP_KV':>7}  {'ms':>8}  {'TFLOPS':>8}  {'GB/s est':>9}  correctness")
    print(f"  {'-'*120}")

    rows = []
    for kv_len in kv_lens:
        seq_lens = [kv_len] * batch_size
        q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=DEVICE)
        kv_cache, kv_indptr, kv_indices, kv_last = make_paged_kv(
            seq_lens, num_kv_heads, head_dim, page_size, dtype
        )
        wrapper = make_wrapper(
            kv_indptr,
            kv_indices,
            kv_last,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            backend,
            fixed_split_size,
            disable_split_kv,
        )

        tile = get_tensor_core_decode_tile_params(lambda: wrapper.run(q, kv_cache))
        ms = bench_ms(lambda: wrapper.run(q, kv_cache))
        flops = decode_flops(batch_size, kv_len, num_qo_heads, head_dim)
        tf = tflops(flops, ms)
        gb = estimated_kv_gb(batch_size, kv_len, num_kv_heads, head_dim, dtype_bytes=2)
        gbps = gb / (ms * 1e-3)

        if skip_correctness:
            corr = "skipped"
        else:
            out_fi = wrapper.run(q, kv_cache)
            corr = correctness_check(q, kv_cache, kv_indptr, kv_indices, kv_last,
                                     seq_lens, page_size, out_fi)

        cta_q = tile.get("CTA_TILE_Q", "?")
        cta_kv = tile.get("CTA_TILE_KV", "?")
        mma_kv = tile.get("NUM_MMA_KV", "?")
        wrp_kv = tile.get("NUM_WARPS_KV", "?")

        print(f"  {kv_len:>8}  {str(cta_q):>6}  {str(cta_kv):>7}  {str(mma_kv):>7}  "
              f"{str(wrp_kv):>7}  {ms:>8.4f}  {tf:>8.3f}  {gbps:>9.1f}  {corr}")

        rows.append({
            "label": label,
            "kv_len": kv_len,
            "batch_size": batch_size,
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "dtype": dtype_label(dtype),
            "use_tensor_cores": 1,
            "backend": backend,
            "fixed_split_size": "" if fixed_split_size is None else fixed_split_size,
            "disable_split_kv": int(disable_split_kv),
            **{k: str(v) for k, v in tile.items()},
            "ms": round(ms, 5),
            "tflops": round(tf, 3),
            "gb_per_s_est": round(gbps, 1),
        })

    return rows


def save_csv(rows, dtype):
    csv_path = DATA_DIR / f"decode_tc_results_{dtype_csv_suffix(dtype)}.csv"

    existing = []
    existing_fieldnames = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            existing = list(reader)

    labels = {r["label"] for r in rows}
    existing = [r for r in existing if r.get("label") not in labels]
    merged = existing + rows

    fieldnames = list(dict.fromkeys(
        list(existing_fieldnames) + [k for r in rows for k in r.keys()]
    ))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(merged)

    print(f"\n  결과 저장: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tensor-core batch decode NUM_MMA_KV tile experiment")
    parser.add_argument("--label", type=str, default="baseline", help="실험 식별자")
    parser.add_argument("--num_qo_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--page_size", type=int, default=16)
    parser.add_argument("--backend", type=str, default="fa2")
    parser.add_argument(
        "--dtype",
        choices=["float16", "fp16", "bfloat16", "bf16"],
        default=os.environ.get("DTYPE", "float16"),
    )
    parser.add_argument("--fixed_split_size", type=str, default=None)
    parser.add_argument("--disable_split_kv", action="store_true")
    parser.add_argument(
        "--kv_lens",
        type=str,
        default=os.environ.get("KV_LENS", " ".join(map(str, DEFAULT_KV_LENS))),
        help="공백 또는 콤마로 구분한 KV length 목록",
    )
    parser.add_argument("--skip_correctness", action="store_true")
    args = parser.parse_args()

    rows = run(
        label=args.label,
        num_qo_heads=args.num_qo_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        batch_size=args.batch_size,
        page_size=args.page_size,
        kv_lens=parse_int_list(args.kv_lens),
        backend=args.backend,
        dtype=parse_dtype(args.dtype),
        fixed_split_size=parse_optional_int(args.fixed_split_size),
        disable_split_kv=args.disable_split_kv,
        skip_correctness=args.skip_correctness,
    )
    save_csv(rows, parse_dtype(args.dtype))
    print(SEP)
