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
    # FlashInfer decode wrapper는 contiguous dense KV가 아니라 paged KV cache를 받는다.
    # 각 batch sequence를 page_size 단위로 쪼개고, 마지막 page에 실제 토큰이 몇 개
    # 들어있는지 kv_last에 기록한다.
    pages_per_seq = [(s + page_size - 1) // page_size for s in seq_lens]
    total_pages = sum(pages_per_seq)

    # Layout: [num_pages, K/V=2, page_size, num_kv_heads, head_dim].
    # wrapper 생성 시 "NHD"를 넘기므로 page 내부 토큰 layout은 N,H,D 순서다.
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
        # indptr[b]: b번째 sequence가 사용하는 page index 범위의 시작점.
        # indices_list에는 실제 page id를 순서대로 넣는다. 여기서는 실험용이라
        # 각 sequence에 연속 page를 할당하지만, wrapper API는 임의 page 순서도 지원한다.
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
    # FlashInfer가 plan/run 중 임시 버퍼로 쓰는 workspace.
    # split-k를 켜면 partial states와 merge에도 이 공간이 사용된다.
    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_tensor_cores=True,
        backend=backend,
    )
    # plan()에서 paged KV 구조, head shape, page size, split-k 정책을 넘기면
    # FlashInfer가 내부 scheduler metadata와 JIT kernel 선택 정보를 준비한다.
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
    # correctness check용 helper. FlashInfer 입력인 paged KV를 PyTorch reference가
    # 다루기 쉬운 dense [batch, seq, kv_head, head_dim] 형태로 되돌린다.
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
    # Tensor-core decode 결과를 검증하기 위한 단순 PyTorch attention.
    # q shape: [batch, num_qo_heads, head_dim]
    # k/v shape after reconstruct: [batch, kv_len, num_kv_heads, head_dim]
    k, v = reconstruct_dense_kv(kv_cache, kv_indptr, kv_indices, kv_last, seq_lens, page_size)
    group_size = q.shape[1] // k.shape[2]

    qf = q.float()
    out = torch.empty_like(qf)
    scale = 1.0 / math.sqrt(q.shape[-1])
    for kv_head in range(k.shape[2]):
        # GQA에서는 하나의 KV head가 여러 QO head를 담당한다.
        # kv_head별로 대응되는 Q head group을 골라 attention을 계산한다.
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
    # benchmark 시간을 오염시키지 않도록 timing 이후에 별도 run 결과를 받아 비교한다.
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


def fixed_split_size_from_count(kv_len, page_size, split_k_count):
    # FlashInfer fixed_split_size is page-based. Convert a requested split-k
    # count into the smallest page chunk that yields at most that many chunks.
    if split_k_count is None:
        return None
    if split_k_count <= 0:
        raise ValueError("split_k_count must be positive")
    return max(1, math.ceil(kv_len / (split_k_count * page_size)))


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


def get_split_kv_plan_info(wrapper, kv_len, page_size):
    # FlashInfer does not expose this via a public API. For the current FA2
    # tensor-core decode path, _plan_info follows PrefillPlanInfo::ToVector():
    #   [9]  = kv_chunk_size_ptr_offset in _int_workspace_buffer
    #   [14] = split_kv flag
    # The stored kv_chunk_size has already been multiplied by page_size, so it is
    # token-based. Keep this isolated so version drift is easy to handle.
    try:
        plan_info = getattr(wrapper, "_plan_info")
        int_workspace = getattr(wrapper, "_int_workspace_buffer")
        kv_chunk_size_ptr_offset = int(plan_info[9])
        split_kv = bool(plan_info[14])
        kv_chunk_size_tokens = (
            int_workspace[kv_chunk_size_ptr_offset:kv_chunk_size_ptr_offset + 4]
            .view(torch.int32)
            .cpu()
            .item()
        )
        kv_chunk_size_pages = (
            kv_chunk_size_tokens // page_size
            if kv_chunk_size_tokens > 0 and kv_chunk_size_tokens % page_size == 0
            else ""
        )
        num_chunks_kv = (
            math.ceil(kv_len / kv_chunk_size_tokens)
            if kv_chunk_size_tokens > 0
            else ""
        )
        return {
            "split_kv": int(split_kv),
            "kv_chunk_size_tokens": kv_chunk_size_tokens,
            "kv_chunk_size_pages": kv_chunk_size_pages,
            "num_chunks_kv": num_chunks_kv,
        }
    except Exception as exc:
        return {
            "split_kv": "",
            "kv_chunk_size_tokens": "",
            "kv_chunk_size_pages": "",
            "num_chunks_kv": "",
            "split_kv_plan_error": type(exc).__name__,
        }


def run(label, num_qo_heads, num_kv_heads, head_dim, batch_size, page_size, kv_lens,
        backend, dtype, fixed_split_size=None, disable_split_kv=False,
        target_split_k=None, skip_correctness=False):
    if target_split_k is not None and fixed_split_size is not None:
        raise ValueError("target_split_k and fixed_split_size are mutually exclusive")
    if target_split_k is not None and disable_split_kv:
        raise ValueError("target_split_k cannot be used with disable_split_kv")

    print(SEP)
    print(f"  label={label}")
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {props.name}  SM{props.major}.{props.minor}")
    print(f"  batch={batch_size}  heads={num_qo_heads}/{num_kv_heads}  "
          f"dim={head_dim}  page={page_size}  dtype={dtype_label(dtype)}")
    print(f"  wrapper: BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True, backend={backend})")
    print(f"  fixed_split_size={fixed_split_size}  target_split_k={target_split_k}  disable_split_kv={disable_split_kv}")
    print(f"  kv_len: {summarize_int_list(kv_lens)}")
    print(SEP)

    print(f"\n  {'kv_len':>8}  {'CTA_Q':>6}  {'CTA_KV':>7}  {'MMA_KV':>7}  "
          f"{'WRP_KV':>7}  {'split':>5}  {'req_k':>5}  {'chunk_tok':>9}  {'chunks':>6}  "
          f"{'ms':>8}  {'TFLOPS':>8}  {'GB/s est':>9}  correctness")
    print(f"  {'-'*140}")

    rows = []
    for kv_len in kv_lens:
        # Decode phase를 모델링하므로 query는 batch마다 1 token이고,
        # KV cache length만 kv_len으로 늘려가며 측정한다.
        seq_lens = [kv_len] * batch_size
        q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=DEVICE)
        kv_cache, kv_indptr, kv_indices, kv_last = make_paged_kv(
            seq_lens, num_kv_heads, head_dim, page_size, dtype
        )
        effective_fixed_split_size = (
            fixed_split_size_from_count(kv_len, page_size, target_split_k)
            if target_split_k is not None
            else fixed_split_size
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
            effective_fixed_split_size,
            disable_split_kv,
        )

        split_plan = get_split_kv_plan_info(wrapper, kv_len, page_size)

        # bench_utils는 FlashInfer JIT/kernel name에서 실제 template tile 값을 추출한다.
        # patch_decode_tc.py로 NUM_MMA_KV를 강제했는지 확인하는 관측 지점이다.
        tile = get_tensor_core_decode_tile_params(lambda: wrapper.run(q, kv_cache))

        # CUDA event 기반 warmup/repeat 측정. 정확도 검증은 timed region에 넣지 않는다.
        ms = bench_ms(lambda: wrapper.run(q, kv_cache))

        # TFLOPS와 GB/s는 실험 간 상대 비교용 proxy다. Decode는 긴 KV read가 중요하므로
        # K/V read bandwidth 추정치도 같이 저장한다.
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

        split_kv = split_plan.get("split_kv", "")
        chunk_tok = split_plan.get("kv_chunk_size_tokens", "")
        chunks = split_plan.get("num_chunks_kv", "")

        req_k = "" if target_split_k is None else target_split_k
        print(f"  {kv_len:>8}  {str(cta_q):>6}  {str(cta_kv):>7}  {str(mma_kv):>7}  "
              f"{str(wrp_kv):>7}  {str(split_kv):>5}  {str(req_k):>5}  {str(chunk_tok):>9}  {str(chunks):>6}  "
              f"{ms:>8.4f}  {tf:>8.3f}  {gbps:>9.1f}  {corr}")

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
            "fixed_split_size": "" if effective_fixed_split_size is None else effective_fixed_split_size,
            "requested_fixed_split_size": "" if fixed_split_size is None else fixed_split_size,
            "target_split_k": "" if target_split_k is None else target_split_k,
            "split_k_count": "" if target_split_k is None else target_split_k,
            "disable_split_kv": int(disable_split_kv),
            **split_plan,
            **{k: str(v) for k, v in tile.items()},
            "ms": round(ms, 5),
            "tflops": round(tf, 3),
            "gb_per_s_est": round(gbps, 1),
        })

    return rows


def save_csv(rows, dtype):
    csv_path = DATA_DIR / f"decode_tc_results_{dtype_csv_suffix(dtype)}.csv"

    # 같은 label을 다시 실행하면 기존 rows를 교체한다.
    # sweep을 중간부터 재실행해도 CSV에 중복 label이 쌓이지 않게 하기 위함이다.
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
    parser.add_argument("--target_split_k", type=str, default=None)
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
        target_split_k=parse_optional_int(args.target_split_k),
        skip_correctness=args.skip_correctness,
    )
    save_csv(rows, parse_dtype(args.dtype))
    print(SEP)
