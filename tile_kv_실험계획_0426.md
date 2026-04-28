# CTA_TILE_KV 실험 계획 (2025-04-26)

## 배경

FlashInfer prefill 커널은 KV 방향 타일 크기(`CTA_TILE_KV`)를 런타임에 자동 결정한다.
이 실험에서는 해당 값을 강제 지정해 성능 변화를 측정한다.

```
CTA_TILE_KV = NUM_MMA_KV × NUM_WARPS_KV × 16
```

RTX 3090 (SM86), LLaMA-3 8B 기준 (`CTA_TILE_Q=128`, `NUM_WARPS_KV=1`):

| NUM_MMA_KV | CTA_TILE_KV | 비고 |
|-----------|-------------|------|
| 1 | 16 | 정상 |
| 2 | 32 | 정상 |
| 4 | 64 | 정상 (FlashInfer 기본 선택값) |
| 8 | 128 | 레지스터 초과 — IsInvalid 우회 필요 |

---

## 수정 파일

```
/root/venv/lib/python3.10/site-packages/flashinfer/data/include/flashinfer/attention/prefill.cuh
```

### Step 1 — NUM_MMA_KV 하드코딩 (매 실험마다 값 교체)

수정 위치:

| 라인 | 함수 | 용도 |
|------|------|------|
| 2601 | `BatchPrefillWithPagedKVCacheDispatched` | `test_tile_kv.py` |
| 2475 | `BatchPrefillWithRaggedKVCacheDispatched` | `bench_attention.py` |
| 1644 | `SinglePrefillWithKVCacheDispatched` | single prefill |

각 위치에서 아래와 같이 변경:

```cpp
// 변경 전
DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
    using KTraits = KernelTraits<...>;
    ...
});

// 변경 후
constexpr size_t NUM_MMA_KV = 1;  // ← 실험할 값으로 교체 (1 / 2 / 4 / 8)
{
    using KTraits = KernelTraits<...>;
    ...
}
```

> 닫는 괄호도 `});` → `}` 로 변경할 것

### Step 2 — NUM_MMA_KV=8 전용: IsInvalid 우회

`line 137` 의 레지스터 상한 조건 완화:

```cpp
// 변경 전 (line 137)
(NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256)

// 변경 후
(NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 512)
```

NUM_MMA_KV=8일 때 수식 값 = 256이므로 상한을 512로 올려야 통과한다.
Register spilling이 발생하지만 연산 결과는 정확하다.
NUM_MMA_KV=8 실험 완료 후 256으로 복원할 것.

---

## 실험 절차

```bash
# 1. prefill.cuh 수정 (NUM_MMA_KV 값 지정)

# 2. JIT 캐시 삭제 (이게 "재빌드" 역할)
rm -rf /root/.cache/flashinfer/

# 3. 실행 (첫 실행 시 JIT 자동 컴파일, 수 분 소요)
cd /root/capstone-yonsei
python test_tile_kv.py --label "NUM_MMA_KV=1"

# 4. 값 바꿔가며 반복
# NUM_MMA_KV: 4(baseline) → 1 → 2 → 8
```

> 첫 실행(NUM_MMA_KV=4)이 baseline 저장, 이후 실행에서 자동 비교

---

## 검증

### 정확도
`test_tile_kv.py`의 `correctness_check()`가 baseline 대비 `max_abs_err` 출력.
fp16 기준 `1e-2` 이하면 정상.

### 실제 적용 확인 (커널명 파싱)
`get_prefill_tile_params()`가 torch profiler로 실행된 커널명에서
`NUM_MMA_KV`, `CTA_TILE_KV` 값을 직접 추출해 출력 → 수정이 실제로 반영됐는지 확인 가능.

### Nsight Compute (심화)
```bash
ncu --metrics sm__shared_memory_load_transactions_per_request \
    --target-processes all \
    python test_tile_kv.py --label "NUM_MMA_KV=1"
```
CTA_TILE_KV가 클수록 shared memory 트랜잭션이 늘어남을 확인.

---

## 결과 저장

- 레이턴시/TFLOPS: `results/tile_kv_results.csv` (자동 저장)
- Baseline 텐서: `results/baselines/seq{N}.pt`