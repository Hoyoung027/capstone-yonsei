"""
CTA_TILE_KV 변경 실험 — BatchPrefillWithPagedKVCacheWrapper

실행:
    # 기본 (FlashInfer 기본 커널, baseline 수집용)
    python test_tile_kv.py --label "baseline"

    # 파라미터 변경 예시
    python test_tile_kv.py --label "baseline_gqa" \\
        --num_qo_heads 32 --num_kv_heads 8 --head_dim 128 \\
        --batch_size 8 --page_size 16

출력:
    - 실제 커널에서 추출한 CTA_TILE_KV / NUM_MMA_KV
    - seq_len 128~8192 (128 간격) 별 레이턴시(ms) / TFLOPS
    - 정확도: baseline 대비 max_abs_err
    - 결과 CSV: results/data/tile_kv_results.csv
"""

import argparse
import csv
import pathlib

import torch
import flashinfer

from bench_attention import get_prefill_tile_params, bench_ms, attention_flops, tflops

RESULTS_DIR  = pathlib.Path(__file__).parent / "results"
DATA_DIR     = RESULTS_DIR / "data"
BASELINE_DIR = RESULTS_DIR / "baselines"
DATA_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LENS = list(range(128, 8193, 128))  # 128, 256, ..., 8192 (64 points)

DEVICE = "cuda"
DTYPE  = torch.float16
SEP    = "=" * 105


# ── paged KV cache 생성 ───────────────────────────────────────

def make_paged_kv(seq_lens, num_kv_heads, head_dim, page_size):
    pages_per_seq = [(s + page_size - 1) // page_size for s in seq_lens]
    total_pages   = sum(pages_per_seq)

    paged_kv_cache = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=DTYPE, device=DEVICE,
    )

    indptr        = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=DEVICE)
    indices_list  = []
    last_page_len = []
    page_offset   = 0
    for i, (s, n_pages) in enumerate(zip(seq_lens, pages_per_seq)):
        indptr[i + 1] = indptr[i] + n_pages
        indices_list.extend(range(page_offset, page_offset + n_pages))
        last_page_len.append(s - (n_pages - 1) * page_size)
        page_offset += n_pages

    return (
        paged_kv_cache,
        indptr,
        torch.tensor(indices_list,  dtype=torch.int32, device=DEVICE),
        torch.tensor(last_page_len, dtype=torch.int32, device=DEVICE),
    )


def make_wrapper(qo_indptr, kv_indptr, kv_indices, kv_last,
                 num_qo_heads, num_kv_heads, head_dim, page_size):
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper   = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        qo_indptr, kv_indptr, kv_indices, kv_last,
        num_qo_heads, num_kv_heads, head_dim, page_size,
    )
    return wrapper


# ── 정확도 검증 ───────────────────────────────────────────────

def correctness_check(seq_len, num_qo_heads, num_kv_heads, head_dim, batch_size, page_size):
    tag = f"seq{seq_len}_h{num_qo_heads}_{num_kv_heads}_d{head_dim}_b{batch_size}_p{page_size}"
    baseline_path = BASELINE_DIR / f"{tag}.pt"

    torch.manual_seed(42)
    seq_lens  = [seq_len] * batch_size
    qo_indptr = torch.arange(
        0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=DEVICE
    )
    q = torch.randn(batch_size * seq_len, num_qo_heads, head_dim, dtype=DTYPE, device=DEVICE)
    torch.manual_seed(42)
    kv_cache, kv_indptr, kv_indices, kv_last = make_paged_kv(seq_lens, num_kv_heads, head_dim, page_size)

    wrapper = make_wrapper(qo_indptr, kv_indptr, kv_indices, kv_last,
                           num_qo_heads, num_kv_heads, head_dim, page_size)
    out = wrapper.run(q, kv_cache).cpu()

    if not baseline_path.exists():
        torch.save(out, baseline_path)
        return "baseline saved"

    baseline = torch.load(baseline_path, weights_only=True, map_location="cpu")
    max_err  = (out - baseline).abs().max().item()
    mean_err = (out - baseline).abs().mean().item()
    status   = "OK" if max_err < 1e-2 else "FAIL"
    return f"max_err={max_err:.5f}  mean_err={mean_err:.5f}  [{status}]"


# ── 메인 실험 ─────────────────────────────────────────────────

def run(label, num_qo_heads, num_kv_heads, head_dim, batch_size, page_size):
    print(SEP)
    print(f"  label={label}")
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {props.name}  SM{props.major}.{props.minor}")
    print(f"  batch={batch_size}  heads={num_qo_heads}/{num_kv_heads}  "
          f"dim={head_dim}  page={page_size}  dtype=fp16")
    print(f"  seq_len: {SEQ_LENS[0]}~{SEQ_LENS[-1]} (step {SEQ_LENS[1]-SEQ_LENS[0]}, {len(SEQ_LENS)} points)")
    print(SEP)

    print(f"\n  {'seq_len':>8}  {'CTA_Q':>6}  {'CTA_KV':>7}  {'MMA_KV':>7}  "
          f"{'WRP_KV':>7}  {'ms':>8}  {'TFLOPS':>8}  correctness")
    print(f"  {'-'*105}")

    rows = []
    for seq_len in SEQ_LENS:
        seq_lens_list = [seq_len] * batch_size
        qo_indptr = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=DEVICE
        )
        q = torch.randn(batch_size * seq_len, num_qo_heads, head_dim, dtype=DTYPE, device=DEVICE)
        kv_cache, kv_indptr, kv_indices, kv_last = make_paged_kv(
            seq_lens_list, num_kv_heads, head_dim, page_size
        )
        wrapper = make_wrapper(qo_indptr, kv_indptr, kv_indices, kv_last,
                               num_qo_heads, num_kv_heads, head_dim, page_size)

        tile  = get_prefill_tile_params(lambda: wrapper.run(q, kv_cache))
        ms    = bench_ms(lambda: wrapper.run(q, kv_cache))
        flops = attention_flops(seq_len, seq_len, num_qo_heads, head_dim, causal=True) * batch_size
        tf    = tflops(flops, ms)
        corr  = correctness_check(seq_len, num_qo_heads, num_kv_heads, head_dim, batch_size, page_size)

        cta_q  = tile.get("CTA_TILE_Q",  "?")
        cta_kv = tile.get("CTA_TILE_KV", "?")
        mma_kv = tile.get("NUM_MMA_KV",  "?")
        wrp_kv = tile.get("NUM_WARPS_KV","?")

        print(f"  {seq_len:>8}  {str(cta_q):>6}  {str(cta_kv):>7}  {str(mma_kv):>7}  "
              f"{str(wrp_kv):>7}  {ms:>8.3f}  {tf:>8.3f}  {corr}")

        rows.append({
            "label":        label,
            "seq_len":      seq_len,
            "batch_size":   batch_size,
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim":     head_dim,
            "page_size":    page_size,
            **{k: str(v) for k, v in tile.items()},
            "ms":           round(ms, 4),
            "tflops":       round(tf, 3),
        })

    return rows


# ── CSV 저장 ──────────────────────────────────────────────────

def save_csv(rows):
    csv_path = DATA_DIR / "tile_kv_results.csv"

    existing = []
    existing_fieldnames = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            existing = list(reader)

    labels = {r["label"] for r in rows}
    existing = [r for r in existing if r.get("label") not in labels]
    merged   = existing + rows

    fieldnames = list(dict.fromkeys(
        list(existing_fieldnames) + [k for r in rows for k in r.keys()]
    ))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(merged)

    print(f"\n  결과 저장: {csv_path}")


# ── 진입점 ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTA_TILE_KV 실험")
    parser.add_argument("--label",        type=str, default="baseline", help="실험 식별자")
    parser.add_argument("--num_qo_heads", type=int, default=32,  help="Query/Output head 수")
    parser.add_argument("--num_kv_heads", type=int, default=8,   help="KV head 수 (GQA)")
    parser.add_argument("--head_dim",     type=int, default=128, help="Head dimension")
    parser.add_argument("--batch_size",   type=int, default=8,   help="Batch size")
    parser.add_argument("--page_size",    type=int, default=16,  help="Paged KV cache page size")
    args = parser.parse_args()

    rows = run(
        label        = args.label,
        num_qo_heads = args.num_qo_heads,
        num_kv_heads = args.num_kv_heads,
        head_dim     = args.head_dim,
        batch_size   = args.batch_size,
        page_size    = args.page_size,
    )
    save_csv(rows)
    print(SEP)
