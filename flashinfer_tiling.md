# FlashInfer 타일링(Tiling) 설정 원리

FlashInfer의 Prefill 커널은 `utils.cuh`와 `attention/prefill.cuh` 두 파일에 걸쳐
타일링 파라미터를 결정한다. 이 문서는 결정 흐름을 단계별로 설명한다.

---

## 0. API 호출 전체 흐름

`flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)` 호출부터
실제 CUDA 커널 런치까지의 전체 경로를 단계별로 설명한다.

```
Python
  flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
      │  flashinfer/prefill.py:1103  (@flashinfer_api 데코레이터)
      │
      ├─ [전처리]
      │    sm_scale = 1/√head_dim
      │    tmp = _get_cache_buf(32MB)        # split-KV용 임시 버퍼
      │    mask_mode = MaskMode.CAUSAL       # causal=True 이므로
      │    backend = determine_attention_backend(...)  # "fa2" 또는 "fa3"
      │
      ├─ get_single_prefill_module(backend, dtype_q, dtype_kv, dtype_o,
      │                            head_dim_qk, head_dim_vo, ...)
      │    │  flashinfer/prefill.py:328  (@functools.cache — 동일 파라미터면 재사용)
      │    │
      │    └─ gen_single_prefill_module(backend, ...)
      │         │  flashinfer/jit/attention/modules.py:489
      │         │
      │         │  [JIT 소스 목록 구성]
      │         │    single_prefill.cu            ← C++ 런처
      │         │    single_prefill_jit_binding.cu ← TVM-FFI 익스포트
      │         │    single_prefill_kernel_mask_0.cu  ┐
      │         │    single_prefill_kernel_mask_1.cu  │ jinja 템플릿으로
      │         │    single_prefill_kernel_mask_2.cu  │ 생성된 explicit
      │         │    single_prefill_kernel_mask_3.cu  ┘ template instantiation
      │         │
      │         └─ .build_and_load()
      │              ninja 빌드 → .so → TVM-FFI로 로드
      │              캐시: ~/.cache/flashinfer/0.6.7/<arch>/cached_ops/<uri>/
      │
      └─ module.run(q, k, v, tmp, out, lse, mask_mode, layout, window_left, ...)

──────────────────────────────────────────────────────────────────────
TVM-FFI (언어 경계)
  module.run(...)  →  TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, single_prefill_with_kv_cache)
                       csrc/single_prefill_jit_binding.cu:28

──────────────────────────────────────────────────────────────────────
C++ 런처
  single_prefill_with_kv_cache(TensorView q, k, v, tmp, o, ...)
      │  csrc/single_prefill.cu:37
      │
      ├─ [shape / stride 추출]
      │    qo_len     = q.size(0)
      │    num_qo_heads = q.size(1)
      │    head_dim_qk  = q.size(2)
      │    kv_len, num_kv_heads = k.size(...)  (kv_layout에 따라)
      │
      ├─ DISPATCH_context(DTypeQ, DTypeKV, DTypeO, IdType,
      │                   MASK_MODE, HEAD_DIM_QK, HEAD_DIM_VO,
      │                   POS_ENCODING_MODE, ..., AttentionVariant, Params)
      │    │  single_prefill_config.inc 에 정의된 매크로
      │    │  런타임 dtype/mask_mode 값을 컴파일 타임 타입으로 변환
      │    │
      │    └─ lambda [&] {
      │         Params params;        // 포인터 + shape + stride 채우기
      │         params.q = ...
      │         params.window_left = window_left;
      │         params.partition_kv = false;
      │
      │         SinglePrefillWithKVCacheDispatched<
      │             HEAD_DIM_QK, HEAD_DIM_VO,
      │             POS_ENCODING_MODE, USE_FP16_QK_REDUCTION,
      │             MASK_MODE, AttentionVariant>(
      │             params, tmp.data_ptr(), stream);
      │       }

──────────────────────────────────────────────────────────────────────
핵심 Dispatch 함수
  SinglePrefillWithKVCacheDispatched<HEAD_DIM_QK, HEAD_DIM_VO, ...>(params, tmp, stream)
      │  include/flashinfer/attention/prefill.cuh:1589
      │
      │  ※ 여기서부터 타일링 파라미터가 결정된다 (Section 1~6 참조)
      │
      ├─ packed_qo_len = qo_len × group_size
      ├─ cta_tile_q = FA2DetermineCtaTileQ(packed_qo_len, HEAD_DIM_VO)
      │                                                           ↑ Section 3
      ├─ DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {
      │    NUM_WARPS_Q  = get_num_warps_q(CTA_TILE_Q)            ↑ Section 4
      │    NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q)
      │    NUM_MMA_Q    = get_num_mma_q(CTA_TILE_Q)
      │    NUM_MMA_D_QK = HEAD_DIM_QK / 16                       ↑ Section 5
      │    NUM_MMA_D_VO = HEAD_DIM_VO / 16
      │
      │    max_num_mma_kv_smem = (smem_budget - Q_smem) / KV_smem_per_tile ↑ Section 6.1
      │    max_num_mma_kv_reg  = (RoPE+큰 VO 조건) ? 2 : 8/NUM_MMA_Q      ↑ Section 6.2
      │
      │    DISPATCH_NUM_MMA_KV(min(smem_lim, reg_lim), NUM_MMA_KV, {        ↑ Section 6.3
      │      KTraits = KernelTraits<MASK_MODE, CTA_TILE_Q, NUM_MMA_Q,
      │                             NUM_MMA_KV, NUM_MMA_D_QK, NUM_MMA_D_VO,
      │                             NUM_WARPS_Q, NUM_WARPS_KV, ...>
      │
      │      if KTraits::IsInvalid() → 에러                                 ↑ Section 7
      │
      │      [Split-KV 결정]                                                ↑ Section 10
      │      max_num_kv_chunks = (occupancy * num_sm) / num_Q_ctas
      │      chunk_size = max(kv_len / max_num_kv_chunks, 256)
      │
      │      if num_chunks <= 1:
      │        → 단일 패스 커널 런치
      │           dim3 nblks(ceil(qo_len*group_size/CTA_TILE_Q), 1, num_kv_heads)
      │           dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV)
      │           cudaLaunchKernel(SinglePrefillWithKVCacheKernel<KTraits>, ...)
      │      else:
      │        → Split-KV 패스
      │           nblks = (Q_tiles, num_chunks, num_kv_heads)
      │           cudaLaunchKernel(...)  // 청크별 partial attention
      │           MergeStates(tmp, lse, o, ...)  // LSE 기반 합산
      │    })
      │  })
      │
      └─ return cudaSuccess

──────────────────────────────────────────────────────────────────────
GPU 커널
  SinglePrefillWithKVCacheKernel<KTraits, Params><<<nblks, nthrs, smem>>>
      include/flashinfer/attention/prefill.cuh
      → Flash Attention 2 메인 루프 실행
```

### 핵심 파일 요약

| 계층 | 파일 | 역할 |
|---|---|---|
| Python API | `flashinfer/prefill.py` | 파라미터 검증, JIT 모듈 로드, TVM-FFI 호출 |
| JIT 빌드 | `flashinfer/jit/attention/modules.py` | ninja 소스 목록 구성, 컴파일, 캐시 관리 |
| TVM-FFI | `csrc/single_prefill_jit_binding.cu` | C++ 함수를 Python에 노출 |
| C++ 런처 | `csrc/single_prefill.cu` | 텐서 shape/stride 추출, dtype dispatch, Params 채우기 |
| 핵심 dispatch | `include/flashinfer/attention/prefill.cuh` | 타일 파라미터 결정, SMEM/레지스터 계산, 커널 런치 |
| 유틸 | `include/flashinfer/utils.cuh` | `FA2DetermineCtaTileQ`, `DISPATCH_*` 매크로 |

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

FlashInfer FA2 커널의 threadBlock은 항상 128 threads = 4 warps로 고정
```
128 threads는 attention 커널에서 경험적으로 검증된 sweet spot

warp를 더 늘리면 thread 수가 늘어 레지스터를 더 나눠야 하므로 per-thread 레지스터 수 감소 → IsInvalid() 조건 초과 가능성 증가
warp를 줄이면 SMEM 로딩 병렬도가 떨어져 메모리 대역폭을 충분히 활용 못함
4 warps = 128 threads는 캐시라인(128B)과 warp-level 메모리 접근이 자연스럽게 맞아떨어지는 크기
```
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

### Q-warp와 KV-warp가 실제로 하는 일

커널 런치 시 warp는 3차원으로 배치된다.

```cpp
dim3 nthrs(32, NUM_WARPS_Q, NUM_WARPS_KV);
cudaLaunchKernel(kernel, nblks, nthrs, ...);
// x: 32 threads (warp 내 lane index, 고정)
// y: NUM_WARPS_Q개 warp (Q 방향)
// z: NUM_WARPS_KV개 warp (KV 방향)
```

하드웨어는 이 구분을 모른다. 커널 코드가 thread index(`warp_idx_y`, `warp_idx_z`)를 읽어 각 warp에게 다른 데이터를 할당한다.

**CTA_TILE_Q=128, NUM_WARPS_Q=4, NUM_WARPS_KV=1인 경우 — 계산 병렬화**

```
Q tile (128행)
┌──────────────┐
│  warp_q=0    │  ← 0~31행 담당
│  warp_q=1    │  ← 32~63행 담당
│  warp_q=2    │  ← 64~95행 담당
│  warp_q=3    │  ← 96~127행 담당
└──────────────┘
```

4개 Q-warp가 각자 다른 Q 행에 대해 독립적으로 Q×K^T와 attention×V를 계산한다. 행이 독립적이므로 warp 간 동기화가 불필요하다.

**CTA_TILE_Q=16, NUM_WARPS_Q=1, NUM_WARPS_KV=4인 경우 — 메모리 대역폭 활용**

Q 행이 16개뿐이라 1 warp로 계산이 충분하다. 남은 3 warp를 버리는 대신 KV 방향으로 돌려 KV tile을 SMEM으로 협력 로드한다.

```
KV tile 로딩
┌──────────────────────────────────────────┐
│ warp_kv=0 │ warp_kv=1 │ warp_kv=2 │ warp_kv=3 │
│ 1/4 구간   │ 1/4 구간   │ 1/4 구간   │ 1/4 구간   │
└──────────────────────────────────────────┘
```

4개 KV-warp가 협력하여 KV tile을 로드하므로 1 warp 단독 대비 4배 빠르다. 로드 후 Q-warp가 계산에 사용한다. 단, 로드 완료 후 `__syncthreads()`로 동기화가 필요하다.

| | Q-warp | KV-warp |
|---|---|---|
| 역할 | 다른 Q 행을 각자 독립 계산 | KV 데이터를 협력해서 SMEM에 로드 |
| 목적 | 계산 병렬화 | 메모리 대역폭 활용 |
| 동기화 | 불필요 (행이 독립) | 필요 (하나의 tile을 나눠 로드) |
| 활성 시점 | Q tile이 클 때 (64, 128) | Q tile이 작을 때 (16) |

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

## 10. 타일 크기의 하드웨어 제약과 유효 범위

### 근본 불변 조건

CTA_TILE_Q와 CTA_TILE_KV는 모두 아래 관계를 만족해야 한다.

```
CTA_TILE_Q  = NUM_WARPS_Q  × NUM_MMA_Q  × 16
CTA_TILE_KV = NUM_WARPS_KV × NUM_MMA_KV × 16
```

**이 조건이 존재하는 이유:** NVIDIA Tensor Core MMA 명령(`mma.sync.aligned.m16n8k16`)은 M 방향으로 최소 16행을 처리한다. 하드웨어에 고정된 명령어 형식이므로 소프트웨어로 변경할 수 없다. 커널 내 각 warp는 `NUM_MMA_*`번의 MMA를 수행하고, `NUM_WARPS_*`개의 warp가 해당 차원을 분담한다. 이 곱이 CTA 타일 크기와 일치하지 않으면 각 warp가 담당하는 행 수가 MMA 최소 단위의 배수가 되지 않아 **silently wrong result**가 발생한다. 에러 없이 실행되지만 결과가 틀린다.

추가로 총 warp 수는 4로 고정되어 있다.

```
NUM_WARPS_Q × NUM_WARPS_KV = 4
```

### CTA_TILE_Q 유효 범위

16의 배수만 유효하다. 4나 8 같은 값은 MMA 최소 단위(16)보다 작으므로 원리적으로 불가능하다.

| CTA_TILE_Q | NUM_WARPS_Q | NUM_WARPS_KV | NUM_MMA_Q | 불변 조건 | 비고 |
|---|---|---|---|---|---|
| 16 | 1 | 4 | 1 | 1×1×16=16 ✓ | 현재 지원 |
| 32 | 2 | 2 | 1 | 2×1×16=32 ✓ | 추가 가능 (아래 참조) |
| 64 | 4 | 1 | 1 | 4×1×16=64 ✓ | 현재 지원 |
| 128 | 4 | 1 | 2 | 4×2×16=128 ✓ | 현재 지원 |
| 256 | 4 | 1 | 4 | 4×4×16=256 — | 레지스터 한계로 실질 불가 |

256이 불가능한 이유: `IsInvalid()` 레지스터 조건에서 head_dim=128 기준으로 계산하면 NUM_MMA_KV가 어떤 값이어도 경계를 초과한다.

```
NUM_MMA_Q=4, NUM_MMA_D_VO=8(head_dim=128), sizeof(DTypeQKAccum)=4:
NUM_MMA_KV=1: 4×(8×8 + 2×4×1) = 4×72 = 288 ≥ 256 → Invalid
```

### CTA_TILE_Q=32 추가 시 수정 필요 파일

현재 코드는 `get_num_warps_q(32) = 4`를 반환해 불변 조건을 위반한다(`4×1×16=64≠32`). 다음 네 파일을 수정해야 한다.

| 파일 | 수정 내용 | 수정하지 않으면 |
|---|---|---|
| `include/flashinfer/utils.cuh:113` | `DISPATCH_CTA_TILE_Q`에 `case 32:` 추가 | 런타임 에러 |
| `include/flashinfer/attention/prefill.cuh:55` | `get_num_warps_q`에 `cta_tile_q > 16 → return 2` 추가 | Silently wrong result |
| `include/flashinfer/utils.cuh:384` | `FA2DetermineCtaTileQ`에 32 반환 조건 추가 | 실행 경로가 32 커널을 타지 않음 |
| `csrc/batch_prefill_paged_kernel_inst.jinja` 등 | 루프에 32 추가 | 링크 에러 |

`get_num_warps_q` 수정이 핵심이다. 이것 없이 나머지만 바꾸면 에러 없이 실행되지만 결과가 틀린다.

---

## 11. CTA_TILE_KV 실험 방법

CTA_TILE_KV는 CTA_TILE_Q와 달리 직접 선택하는 값이 아니다. SMEM과 레지스터 제약 안에서 **자동으로 최대화**된다. 또한 `NUM_WARPS_KV = 4 / NUM_WARPS_Q`로 CTA_TILE_Q에 종속되어 있어 독립적으로 설정할 수 없다.

실험적으로 줄이려면 `prefill.cuh`의 `DISPATCH_NUM_MMA_KV` 호출부에서 `min()` 결과를 강제로 덮어쓰는 방식을 쓴다.

```cpp
// 현재: 제약 안에서 최대값 자동 선택
DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, { ... })

// 실험: NUM_MMA_KV를 2로 강제
DISPATCH_NUM_MMA_KV(2, NUM_MMA_KV, { ... })
```

이 경우 `CTA_TILE_KV = 2 × NUM_WARPS_KV × 16`이 된다. CTA_TILE_Q=64일 때 NUM_WARPS_KV=1이므로 CTA_TILE_KV=32가 된다.

---

## 12. Split-KV (KV 파티셔닝) (single prefill 기준)

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

## 13. 런타임 상수 → 컴파일 타임 상수 변환 패턴

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

## 14. 파라미터 결정 트리 요약

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
