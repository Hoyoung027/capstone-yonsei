# FlashInfer 타일링(Tiling) 설정 원리

FlashInfer의 Prefill 커널은 `utils.cuh`와 `attention/prefill.cuh` 두 파일에 걸쳐
타일링 파라미터를 결정한다. 이 문서는 결정 흐름을 단계별로 설명한다.

---

## 1. 타일링 파라미터 전체 목록

| 파라미터 | 의미 | 단위 |
|---|---|---|
| `CTA_TILE_Q` | 한 CTA(블록)이 담당하는 Query 토큰 수 | 토큰 |
| `CTA_TILE_KV` | 한 CTA가 한 번에 처리하는 KV 토큰 수 | 토큰 |
| `NUM_MMA_Q` | Q 방향 MMA 타일 수 (warp 당) | 개 |
| `NUM_MMA_KV` | KV 방향 MMA 타일 수 (warp 당) | 개 |
| `NUM_MMA_D_QK` | QK head_dim 방향 MMA 타일 수 | 개 |
| `NUM_MMA_D_VO` | VO head_dim 방향 MMA 타일 수 | 개 |
| `NUM_WARPS_Q` | Q 방향 warp 수 | 개 |
| `NUM_WARPS_KV` | KV 방향 warp 수 | 개 |

모든 파라미터는 `KernelTraits` 구조체 하나로 묶여 커널 전체에 전달된다.

---

## 2. 결정 흐름 요약

```
입력: qo_len, kv_len, head_dim, num_qo_heads, num_kv_heads
         │
         ▼
[Step 1] CTA_TILE_Q 결정  ─── FA2DetermineCtaTileQ()  (utils.cuh)
         │
         ▼
[Step 2] NUM_WARPS_Q / NUM_WARPS_KV / NUM_MMA_Q 결정  (prefill.cuh)
         │
         ▼
[Step 3] NUM_MMA_D_QK / NUM_MMA_D_VO 결정  ─── HEAD_DIM / 16
         │
         ▼
[Step 4] NUM_MMA_KV 결정  ─── SMEM & 레지스터 제약
         │
         ▼
[Step 5] KernelTraits 인스턴스화 + IsInvalid() 검증
         │
         ▼
[Step 6] CTA_TILE_KV 계산  ─── NUM_MMA_KV * NUM_WARPS_KV * 16
```

---

## 3. Step 1 — CTA_TILE_Q 결정

**파일:** `include/flashinfer/utils.cuh`, 384번째 줄

`FA2DetermineCtaTileQ()`는 런타임에 `avg_packed_qo_len`과 `head_dim`을 보고
CTA가 처리할 Query 타일 크기(128 / 64 / 16)를 선택한다.

```cpp
inline uint32_t FA2DetermineCtaTileQ(int64_t avg_packed_qo_len, uint32_t head_dim) {
  if (avg_packed_qo_len > 64 && head_dim < 256) {
    return 128;   // 긴 시퀀스 + 작은 head_dim → 큰 타일로 parallelism 확보
  } else {
    auto compute_capacity = GetCudaComputeCapability();
    if (compute_capacity.first >= 8) {   // Ampere 이상
      if (avg_packed_qo_len > 16) {
        return 64;                        // 중간 길이
      } else {
        return 16;                        // decode처럼 매우 짧은 경우
      }
    } else {
      // Turing(sm75): 1x4 warp 레이아웃을 지원할 shared memory가 부족
      return 64;
    }
  }
}
```

### avg_packed_qo_len 계산

`avg_packed_qo_len = qo_len * group_size` (group_size = num_qo_heads / num_kv_heads)

GQA(Grouped Query Attention) 환경에서는 KV 헤드 당 처리해야 할 논리적 Query 토큰 수가
`group_size`배 증가하므로 packed 개념을 사용한다.

### 선택 기준 정리

| avg_packed_qo_len | head_dim | GPU | CTA_TILE_Q |
|---|---|---|---|
| > 64 | < 256 | Any | **128** |
| 17 ~ 64 | ≥ 256 또는 any | Ampere+ | **64** |
| ≤ 16 | any | Ampere+ | **16** |
| any | any | Turing (sm75) | **64** |

---

## 4. Step 2 — Warp 레이아웃과 NUM_MMA_Q

**파일:** `include/flashinfer/attention/prefill.cuh`, 55~73번째 줄

CTA 내부 warp는 Q 방향과 KV 방향의 2D 격자로 배치된다.
총 warp 수는 항상 4로 고정된다.

```cpp
// Q 방향 warp 수: CTA_TILE_Q > 16이면 4, 아니면 1
constexpr uint32_t get_num_warps_q(const uint32_t cta_tile_q) {
  if (cta_tile_q > 16) {
    return 4;
  } else {
    return 1;
  }
}

// KV 방향 warp 수: 총 4 warp를 Q 방향이 나눠 가진 나머지
constexpr uint32_t get_num_warps_kv(const uint32_t cta_tile_kv) {
  return 4 / get_num_warps_q(cta_tile_kv);
}

// warp 당 Q 방향 MMA 타일 수
constexpr uint32_t get_num_mma_q(const uint32_t cta_tile_q) {
  if (cta_tile_q > 64) {
    return 2;
  } else {
    return 1;
  }
}
```

### CTA_TILE_Q별 warp 레이아웃 정리

| CTA_TILE_Q | NUM_WARPS_Q | NUM_WARPS_KV | NUM_MMA_Q |
|---|---|---|---|
| 128 | 4 | 1 | 2 |
| 64 | 4 | 1 | 1 |
| 16 | 1 | 4 | 1 |

- `CTA_TILE_Q=128`이면 Q 방향 4 warp × 2 MMA = 16×2×4 = 128 토큰
- `CTA_TILE_Q=16`이면 Q 방향 1 warp × 1 MMA = 16 토큰, KV 방향으로 4 warp를 활용

---

## 5. Step 3 — HEAD_DIM 방향 MMA 수

MMA 연산 하나의 크기는 16×16이므로 head_dim을 16으로 나누면 MMA 타일 수가 된다.

```cpp
// prefill.cuh 내 Dispatched 함수들에서
constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
```

### 예시

| HEAD_DIM | NUM_MMA_D |
|---|---|
| 64 | 4 |
| 128 | 8 |
| 256 | 16 |
| 512 | 32 |

---

## 6. Step 4 — NUM_MMA_KV 결정 (핵심)

**파일:** `include/flashinfer/attention/prefill.cuh`의 `SinglePrefillWithKVCacheDispatched()`
및 `BatchPrefillWithRaggedKVCacheDispatched()` 등

NUM_MMA_KV는 두 가지 상한값의 최솟값으로 결정된다.

### 6.1 SMEM 기반 상한 (max_num_mma_kv_smem)

```cpp
// SM 당 최대 shared memory 확인
int max_smem_per_sm = 0;
cudaDeviceGetAttribute(&max_smem_per_sm,
                       cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id);

// SM 당 2개 CTA 동시 실행 가능한지 판단
const int num_ctas_per_sm =
    max_smem_per_sm >= 2 * (CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ) +
                            (HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV))
        ? 2 : 1;

const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

// KV smem 예산 = 전체 smem - Q smem
// KV tile 하나의 크기 = (HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV)
const uint32_t max_num_mma_kv_smem =
    (max_smem_per_threadblock - CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ)) /
    ((HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV));
```

**SMEM 레이아웃** (`SharedStorageQKVO` 구조체, `prefill.cuh` 75번째 줄):

```
SharedMemory (union)
├── [KQV 로딩용]
│   ├── q_smem  : CTA_TILE_Q × HEAD_DIM_QK  × sizeof(DTypeQ)
│   ├── k_smem  : CTA_TILE_KV × HEAD_DIM_QK × sizeof(DTypeKV)
│   └── v_smem  : CTA_TILE_KV × HEAD_DIM_VO × sizeof(DTypeKV)
├── [CTA 동기화용 — NUM_WARPS_KV > 1인 경우]
│   ├── cta_sync_o_smem  : NUM_WARPS_KV × CTA_TILE_Q × HEAD_DIM_VO × sizeof(float)
│   └── cta_sync_md_smem : NUM_WARPS_KV × CTA_TILE_Q × sizeof(float2)
└── [출력 임시 버퍼]
    └── smem_o  : CTA_TILE_Q × HEAD_DIM_VO × sizeof(DTypeO)
```

`CTA_TILE_KV = NUM_MMA_KV * NUM_WARPS_KV * 16` 이므로 SMEM 예산이 클수록
더 많은 NUM_MMA_KV를 수용할 수 있다.

### 6.2 레지스터 기반 상한 (max_num_mma_kv_reg)

```cpp
// KernelTraits::IsInvalid()에 반영되는 레지스터 제약
// 한 스레드의 레지스터 사용량 ≈ NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV)
// 이 값이 256 이상이면 Invalid

const uint32_t max_num_mma_kv_reg =
    (HEAD_DIM_VO >= 128                              // head_dim이 크면
     && NUM_MMA_Q == 2                               // Q 타일도 크고
     && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama  // RoPE까지 계산하면
     && !USE_FP16_QK_REDUCTION)
        ? 2                    // 레지스터 압박이 크므로 NUM_MMA_KV 최대 2
        : (8 / NUM_MMA_Q);     // 일반 경우: NUM_MMA_Q=1 → 최대 8, NUM_MMA_Q=2 → 최대 4
```

### 6.3 최종 선택

```cpp
DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, { ... });
```

`DISPATCH_NUM_MMA_KV` 매크로 (`utils.cuh` 94번째 줄)는 런타임 값을
컴파일 타임 상수로 변환한다. 지원하는 값은 8, 4, 2, 1이며 입력값 이하의
최대 2의 거듭제곱을 선택한다.

```cpp
#define DISPATCH_NUM_MMA_KV(max_mma_kv, NUM_MMA_KV, ...) \
  if (max_mma_kv >= 8) {                                 \
    constexpr size_t NUM_MMA_KV = 8;                     \
    __VA_ARGS__                                          \
  } else if (max_mma_kv >= 4) {                          \
    constexpr size_t NUM_MMA_KV = 4;                     \
    __VA_ARGS__                                          \
  } else if (max_mma_kv >= 2) {                          \
    constexpr size_t NUM_MMA_KV = 2;                     \
    __VA_ARGS__                                          \
  } else if (max_mma_kv >= 1) {                          \
    constexpr size_t NUM_MMA_KV = 1;                     \
    __VA_ARGS__                                          \
  } else { FLASHINFER_ERROR(...); }
```

---

## 7. Step 5 — KernelTraits 인스턴스화

**파일:** `include/flashinfer/attention/prefill.cuh`, 95~167번째 줄

모든 타일링 파라미터는 `KernelTraits` 구조체로 묶인다.

```cpp
template <MaskMode MASK_MODE_,
          uint32_t CTA_TILE_Q_,
          uint32_t NUM_MMA_Q_,   uint32_t NUM_MMA_KV_,
          uint32_t NUM_MMA_D_QK_, uint32_t NUM_MMA_D_VO_,
          uint32_t NUM_WARPS_Q_, uint32_t NUM_WARPS_KV_,
          PosEncodingMode POS_ENCODING_MODE_,
          typename DTypeQ_, typename DTypeKV_, typename DTypeO_,
          typename DTypeQKAccum_, typename IdType_, typename AttentionVariant_>
struct KernelTraits {
  // ── 파생 상수 ────────────────────────────────────
  static constexpr uint32_t NUM_THREADS  = NUM_WARPS_Q * NUM_WARPS_KV * WARP_SIZE;
  static constexpr uint32_t HEAD_DIM_QK  = NUM_MMA_D_QK * 16;
  static constexpr uint32_t HEAD_DIM_VO  = NUM_MMA_D_VO * 16;
  static constexpr uint32_t CTA_TILE_KV  = NUM_MMA_KV * NUM_WARPS_KV * 16;

  // ── upcast stride (permuted smem 접근용) ─────────
  static constexpr uint32_t UPCAST_STRIDE_Q = HEAD_DIM_QK / upcast_size<DTypeQ_>();
  static constexpr uint32_t UPCAST_STRIDE_K = HEAD_DIM_QK / upcast_size<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_V = HEAD_DIM_VO / upcast_size<DTypeKV_>();

  // ── Swizzle 모드 ──────────────────────────────────
  // fp8 + head_dim=64는 64B swizzle, 나머지는 128B swizzle
  static constexpr SwizzleMode SWIZZLE_MODE_KV =
      (sizeof(DTypeKV_) == 1 && HEAD_DIM_VO == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;

  // 128B swizzle: 스레드 레이아웃 4행×8열
  // 64B swizzle : 스레드 레이아웃 8행×4열
  static constexpr uint32_t KV_THR_LAYOUT_ROW =
      SWIZZLE_MODE_KV == SwizzleMode::k128B ? 4 : 8;
  static constexpr uint32_t KV_THR_LAYOUT_COL =
      SWIZZLE_MODE_KV == SwizzleMode::k128B ? 8 : 4;

  using SharedStorage = SharedStorageQKVO<...>;
};
```

### IsInvalid() — 유효하지 않은 설정 필터링

```cpp
static constexpr bool IsInvalid() {
  return
    // VO head_dim이 너무 작음 (MMA 최소 단위 미달)
    (NUM_MMA_D_VO < 4) ||
    // VO=4이면서 KV 타일이 홀수 → 레이아웃 대칭성 위반
    (NUM_MMA_D_VO == 4 && NUM_MMA_KV % 2 == 1) ||
    // RoPE + 큰 VO 조합 시 warp 간 분할 불가
    (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama
     && NUM_MMA_D_VO > 4 && NUM_MMA_D_VO % (2 * NUM_WARPS_Q) != 0) ||
    // 레지스터 스필 방지: 스레드 당 레지스터 ≥ 256개 금지
    (NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256) ||
    // fp8 KV: KV 타일 로딩 시 warp 간 균등 분배 제약
    (sizeof(DTypeKV) == 1 && NUM_MMA_KV * 2 % NUM_WARPS_Q != 0) ||
    // fp8 + RoPE: 미지원 조합
    (sizeof(DTypeKV) == 1 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama);
}
```

---

## 8. Step 6 — CTA_TILE_KV 계산

`CTA_TILE_KV`는 앞선 결정들로부터 자동으로 계산된다.

```
CTA_TILE_KV = NUM_MMA_KV × NUM_WARPS_KV × 16
```

**예시:**

| CTA_TILE_Q | NUM_WARPS_KV | NUM_MMA_KV | CTA_TILE_KV |
|---|---|---|---|
| 128 | 1 | 8 | 128 |
| 128 | 1 | 4 | 64 |
| 64 | 1 | 8 | 128 |
| 16 | 4 | 4 | 256 |
| 16 | 4 | 2 | 128 |

---

## 9. 전체 예시: fp16, head_dim=128, Ampere GPU

```
입력:
  qo_len = 512, kv_len = 512, group_size = 1 (MHA)
  head_dim_qk = head_dim_vo = 128
  dtype = fp16, GPU = A100 (sm80)

─────────────────────────────────────────────────
Step 1: CTA_TILE_Q
  avg_packed_qo_len = 512 * 1 = 512 > 64, head_dim = 128 < 256
  → CTA_TILE_Q = 128

Step 2: Warp 레이아웃
  CTA_TILE_Q = 128 > 16 → NUM_WARPS_Q = 4
  NUM_WARPS_KV = 4 / 4 = 1
  CTA_TILE_Q = 128 > 64 → NUM_MMA_Q = 2

Step 3: HEAD_DIM 방향
  NUM_MMA_D_QK = 128 / 16 = 8
  NUM_MMA_D_VO = 128 / 16 = 8

Step 4: NUM_MMA_KV
  A100 max_smem_per_sm = 164KB
  Q smem = 128 * 128 * 2 = 32KB
  KV smem 한 타일 = (128+128) * 16 * 1 * 2 = 8KB
  가용 smem ≈ 82KB - 32KB = 50KB
  max_num_mma_kv_smem = 50KB / 8KB = 6 → 실제 사용 가능한 최대값 4 (≤6인 최대 2의 거듭제곱 아님, 정수 나눗셈이므로 6)
  max_num_mma_kv_reg   = 8 / NUM_MMA_Q = 8 / 2 = 4
  NUM_MMA_KV = min(6, 4) = 4 → DISPATCH_NUM_MMA_KV → 4

Step 5: KernelTraits 확인
  레지스터 = 2 * (8*8 + 2*2*4) = 2 * (64+16) = 160 < 256 ✓

Step 6: CTA_TILE_KV
  CTA_TILE_KV = 4 * 1 * 16 = 64

결과:
  CTA_TILE_Q=128, CTA_TILE_KV=64
  NUM_WARPS_Q=4, NUM_WARPS_KV=1, NUM_THREADS=128
  NUM_MMA_Q=2, NUM_MMA_KV=4
  NUM_MMA_D_QK=8, NUM_MMA_D_VO=8
```

---

## 10. Split-KV (KV 파티셔닝)

타일링 결정 이후, 커널이 충분한 병렬성을 갖는지 판단하여 KV 차원을 쪼갠다.

```cpp
// SinglePrefillWithKVCacheDispatched 내부
int num_blocks_per_sm = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks_per_sm, kernel, num_threads, smem_size);

uint32_t max_num_kv_chunks =
    (num_blocks_per_sm * num_sm) /
    (num_kv_heads * ceil_div(qo_len * group_size, CTA_TILE_Q));

if (max_num_kv_chunks > 1) {
  // chunk_size ≥ 256 되도록 쪼갬
  uint32_t chunk_size = max(ceil_div(kv_len, max_num_kv_chunks), 256);
  num_chunks = ceil_div(kv_len, chunk_size);
}
```

Split-KV를 사용하면 최종적으로 `MergeStates()` (LSE 기반 partial softmax 합산)
또는 `AttentionSum()`으로 청크별 결과를 합친다.

---

## 11. 런타임 상수 → 컴파일 타임 상수 변환 패턴

FlashInfer는 런타임 값을 컴파일 타임 `constexpr`로 변환하기 위해 `DISPATCH_*` 매크로를
중첩해서 사용한다. 이를 통해 컴파일러가 루프 언롤링·레지스터 최적화를 충분히 수행할 수 있다.

```cpp
// utils.cuh의 매크로들이 중첩되는 예시 (SinglePrefillWithKVCacheDispatched)
DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {          // 128 / 64 / 16
  // NUM_WARPS_Q, NUM_WARPS_KV, NUM_MMA_Q는 constexpr 함수로 즉시 결정
  constexpr uint32_t NUM_WARPS_Q  = get_num_warps_q(CTA_TILE_Q);
  constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);
  constexpr uint32_t NUM_MMA_Q    = get_num_mma_q(CTA_TILE_Q);

  DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {  // 8/4/2/1
    using KTraits = KernelTraits<
        MASK_MODE, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV,
        NUM_MMA_D_QK, NUM_MMA_D_VO, NUM_WARPS_Q, NUM_WARPS_KV,
        POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeO,
        DTypeQKAccum, IdType, AttentionVariant>;

    if constexpr (!KTraits::IsInvalid()) {
      // 여기서 비로소 실제 커널 런치
      auto kernel = SinglePrefillWithKVCacheKernel<KTraits, Params>;
      cudaLaunchKernel(...);
    }
  })
})
```

각 `DISPATCH_*` 매크로는 런타임 분기처럼 보이지만, 내부에서 `constexpr` 변수를 선언하므로
실제로는 별도의 커널 인스턴스(템플릿 특수화)가 컴파일 타임에 생성된다.

---

## 12. 파라미터 결정 트리 요약

```
FA2DetermineCtaTileQ(avg_packed_qo_len, head_dim)
    │
    ├─ avg_packed_qo_len > 64 AND head_dim < 256 → CTA_TILE_Q = 128
    │     → NUM_WARPS_Q=4, NUM_WARPS_KV=1, NUM_MMA_Q=2
    │
    ├─ Ampere+ AND avg_packed_qo_len > 16       → CTA_TILE_Q = 64
    │     → NUM_WARPS_Q=4, NUM_WARPS_KV=1, NUM_MMA_Q=1
    │
    ├─ Ampere+ AND avg_packed_qo_len ≤ 16       → CTA_TILE_Q = 16
    │     → NUM_WARPS_Q=1, NUM_WARPS_KV=4, NUM_MMA_Q=1
    │
    └─ Turing (sm75)                            → CTA_TILE_Q = 64
          → NUM_WARPS_Q=4, NUM_WARPS_KV=1, NUM_MMA_Q=1

NUM_MMA_KV = min(
    floor((max_smem_per_threadblock - Q_smem) / KV_smem_per_mma_kv),  ← SMEM 제약
    (HEAD_DIM_VO >= 128 AND NUM_MMA_Q=2 AND RoPE AND !fp16qk) ? 2     ← 레지스터 제약
                                                               : 8/NUM_MMA_Q
)
→ DISPATCH_NUM_MMA_KV: 8 ≥ 4 ≥ 2 ≥ 1 중 선택

CTA_TILE_KV = NUM_MMA_KV × NUM_WARPS_KV × 16
```
