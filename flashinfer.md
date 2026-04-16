# FlashInfer가 하는 일

FlashInfer는 LLM serving 시스템 전체가 아니라, 그 안에서 attention 연산을 빠르게 실행하는 GPU 가속 라이브러리다.

---

## 1. FlashInfer의 역할

FlashInfer는 주로 attention 계산에 필요한 입력을 받아 prefill/decode 결과를 만들어내는 역할을 맡는다.

- 입력
  - `q`
  - `k/v cache`
  - `indptr`
  - `page table`
  - attention mask
  - RoPE, softmax scale, backend 설정
- 출력
  - prefill attention 결과
  - decode attention 결과
  - 필요하면 `lse` 같은 보조 결과

FlashInfer가 직접 하지 않는 일도 분명하다.

- request queue 관리
- batching scheduler 구현
- tokenizer 실행
- KV cache eviction policy 결정
- HTTP 서버나 serving endpoint 운영

즉 FlashInfer는 "요청을 받아 모델 전체를 운영하는 시스템"이 아니라, "모델 내부 attention을 빠르게 계산하는 실행 엔진"에 가깝다.

---

## 2. Serving 시스템 안에서의 위치

LLM serving 시스템은 보통 여러 층으로 나뉜다.

- serving engine: 전체 요청 흐름과 모델 실행을 관리
- scheduler: 어떤 요청들을 한 배치로 묶을지 결정
- model runner: 각 layer의 forward를 호출
- FlashInfer: attention 연산을 실제 GPU kernel로 빠르게 수행

비유하면 serving engine은 식당 전체 운영이고, scheduler는 주문을 어떤 순서로 주방에 보낼지 정하는 사람이다. FlashInfer는 그 안에서 실제 요리를 빠르게 해내는 고성능 조리 장비에 가깝다. 식당 운영 방식은 바뀔 수 있어도, 주방 장비는 주어진 재료와 작업 단위를 받아 빠르게 처리하는 데 집중한다.

---

## 3. 이종 모델 요청이 같은 GPU로 들어올 때

FlashInfer 계층에서 어디까지 대응 가능하고, 어디서부터는 상위 계층이 책임져야 하는지로 나눠서 이해하는 것이 정확하다.

### FlashInfer 계층에서 대응 가능한 것

**JIT 모듈 캐싱 — 모델 아키텍처마다 별도 커널**

`get_batch_prefill_module()`은 `@functools.cache`로 감싸져 있어서, `(dtype, head_dim_qk, head_dim_vo, backend, ...)` 조합마다 별도의 컴파일된 `.so`를 캐싱한다. 처음 등장하는 조합은 JIT 컴파일이 발생하고, 이후에는 디스크 캐시에서 즉시 로드된다. 커널 코드 수준에서는 모델마다 별도로 준비된다.

**plan() — 배치 shape마다 독립적으로 재계산**

`plan()`은 현재 배치의 `qo_indptr`, `kv_len`, `batch_size`, `num_heads` 등에 맞춰 `cta_tile_q`, `kv_chunk_size`, work partition을 매번 새로 계산한다. 모델 A의 배치와 모델 B의 배치가 섞이지 않고 각자 plan된다면, 각각의 shape에 맞게 최적 타일링이 적용된다.

### FlashInfer 계층에서 대응할 수 없는 것

FlashInfer는 "주어진 배치를 빠르게 실행"만 한다. 요청을 어떻게 묶느냐는 전적으로 상위 serving engine의 책임이다.

| 문제 | 왜 FlashInfer가 책임지지 않는가 |
|------|-------------------------------|
| 이종 모델 요청을 같은 배치에 섞을 수 없음 | 모델마다 `head_dim`, `num_kv_heads`, `dtype`이 달라 단일 kernel call 불가 |
| GPU 메모리 분할 | KV cache, workspace buffer 할당은 serving engine이 해야 함 |
| 모델 전환 오버헤드 | 가중치 로딩/교체는 FlashInfer 범위 밖 |
| CUDA stream 격리 | 두 모델이 동시에 GPU를 쓸 때 stream 충돌 방지는 위 계층 책임 |

### 실제 구조

FlashInfer 기반 엔진이 이종 모델을 같은 GPU에서 돌리려면, serving engine 쪽에서 모델별로 wrapper 인스턴스를 분리한다.

```
GPU 메모리
├── 모델 A 가중치
├── 모델 A KV cache
├── 모델 A FlashInfer wrapper  (plan_info A, workspace A)
│
├── 모델 B 가중치
├── 모델 B KV cache
└── 모델 B FlashInfer wrapper  (plan_info B, workspace B)
```

wrapper는 `_plan_info`, `_float_workspace_buffer` 등 상태를 내부적으로 들고 있기 때문에, 모델별로 분리하면 FlashInfer 계층에서는 충돌 없이 동작한다.

| 관점 | FlashInfer의 대응 |
|------|-----------------|
| 다른 dtype/head_dim 모델 | JIT 캐시로 자동 대응 (첫 실행 시 컴파일) |
| 배치마다 다른 시퀀스 분포 | plan()이 매번 최적 타일링 재계산 |
| 이종 모델 동시 서빙 | 직접 지원 안 함 — wrapper를 모델별로 분리해야 함 |
| 요청 스케줄링/라우팅 | FlashInfer 범위 밖 — serving engine이 책임 |

---

## 4. 주요 API 구분

FlashInfer의 prefill 관련 API는 크게 single request용, batch wrapper용, backend-native low-level API로 나뉜다.

| 구분 | 대표 API | 설명 |
|---|---|---|
| Single request | `single_prefill_with_kv_cache` | 단일 요청의 prefill attention |
| Batch + paged KV | `BatchPrefillWithPagedKVCacheWrapper` | 배치 prefill, paged KV cache 사용 |
| Batch + ragged KV | `BatchPrefillWithRaggedKVCacheWrapper` | 배치 prefill, ragged KV tensor 사용 |
| cuDNN direct | `cudnn_batch_prefill_with_kv_cache` | cuDNN backend를 직접 호출 |
| TRT-LLM direct | `trtllm_batch_context_with_kv_cache` | TRT-LLM 스타일 paged context attention 직접 호출 |

보통 batch serving 관점에서 기본 인터페이스는 wrapper API다.

- `BatchPrefillWithPagedKVCacheWrapper.plan(...)`
- `BatchPrefillWithPagedKVCacheWrapper.run(...)`
- `BatchPrefillWithRaggedKVCacheWrapper.plan(...)`
- `BatchPrefillWithRaggedKVCacheWrapper.run(...)`

반면 `cudnn_batch_prefill_with_kv_cache`와 `trtllm_batch_context_with_kv_cache`는 backend를 직접 비교하거나 특정 low-level 경로를 쓸 때 더 가깝다.

---

## 5. Paged vs Ragged

`paged`와 `ragged`는 backend 종류가 아니라 KV cache를 메모리에 저장하는 방식의 차이다.

- ragged
  - KV를 길이 기준으로 이어붙인 varlen tensor
  - 예: `k/v = [total_kv_tokens, num_kv_heads, head_dim]`
  - 추가 메타데이터: `kv_indptr`
- paged
  - KV를 page 단위 블록으로 나눠 저장
  - 예: `k/v = [num_pages, page_size, num_kv_heads, head_dim]` 또는 HND layout
  - 추가 메타데이터: `paged_kv_indptr`, `paged_kv_indices`, `paged_kv_last_page_len`

ragged는 구조가 단순해서 실험하거나 동작을 이해하기 쉽다. 반면 paged는 실제 serving 시스템의 KV cache allocator, block table, prefix cache 구조와 더 잘 맞기 때문에 온라인 serving에서는 더 자연스럽다.

---

## 6. Backend 구분

`fa2/fa3`, `cudnn`, `trtllm-gen`은 "어떤 커널 구현을 쓰느냐"를 뜻한다.

- `fa2`, `fa3`
  - FlashInfer가 주로 사용하는 attention backend
  - FlashAttention 계열 고성능 구현에 가깝다
- `cudnn`
  - cuDNN backend
  - wrapper 경유 또는 direct API로 사용할 수 있다
- `trtllm-gen`
  - TRT-LLM 스타일 backend
  - 특히 paged context attention 경로와 연결된다

**중요한 점은 backend 축과 data layout 축은 서로 다르다는 것이다.**

- `paged` / `ragged`: 데이터를 어떻게 저장하는가
- `fa2` / `fa3` / `cudnn` / `trtllm-gen`: 어떤 backend kernel로 계산하는가

---

## 7. compile / plan / run

FlashInfer의 실행은 보통 `compile -> plan -> run` 세 단계로 이해하면 된다.

- `compile`
  - dtype, head_dim, backend, pos encoding 같은 조합이 처음 등장할 때 JIT build 또는 cached module load가 일어날 수 있다
  - 같은 조합이면 이후에는 보통 캐시된 모듈을 재사용한다
- `plan`
  - 이번 batch의 `indptr`, `seq_lens`, `page table`, batch size에 맞춰 runtime work partition과 tiling 계획을 계산한다
  - wrapper 내부 상태와 `plan_info`를 만들어 여러 layer에서 재사용할 수 있게 한다
- `run`
  - 실제 attention kernel을 실행한다

핵심은 다음 문장으로 요약할 수 있다.

**매 요청마다 compile되는 것은 아니고, 배치가 달라질 때마다 plan은 다시 될 수 있다.**

즉 serving 시스템이 켜져 있어도, 매번 모든 걸 처음부터 다시 하는 구조는 아니다. 커널 종류는 비교적 오래 재사용되고, 현재 batch의 shape와 길이 분포에 맞는 실행 계획만 다시 계산될 수 있다.

---

## 8. 요청 하나가 처리되는 흐름

아래는 batch prefill 기준의 전형적인 흐름이다.

```text
사용자 요청 수신
  -> scheduler가 여러 요청을 batch로 구성
  -> KV cache / indptr / page table 준비
  -> FlashInfer wrapper plan(...)
  -> 각 transformer layer에서 run(...)
  -> attention 결과를 다음 layer로 전달
  -> 최종 출력 반환
```

이 구조 때문에 `plan(...)`은 보통 "현재 배치에 대해 한 번", `run(...)`은 "각 layer에서 반복" 호출되는 형태가 된다.

---

## 9. Flashinfer 폴더에서 중요하게 볼 포인트

이 저장소에서는 [bench_attention.py](/root/capstone-yonsei/bench_attention.py:172)가 single API 중심 벤치를 수행한다. 따라서 지금 벤치 결과를 읽을 때는 `single_prefill_with_kv_cache`가 어떤 JIT module과 kernel 경로를 타는지 보는 것이 핵심이다.

반대로 batch serving 관점에서 FlashInfer를 이해하려면 `flashinfer/flashinfer/prefill.py` 안의 wrapper `plan/run` 흐름을 보는 것이 더 중요하다. 특히 `BatchPrefillWithPagedKVCacheWrapper`와 `BatchPrefillWithRaggedKVCacheWrapper`의 `plan(...)`은 실행 전에 어떤 메타데이터와 work partition을 준비하는지 보여준다.

세부 커널 dispatch와 타일링은 기존 [flashinfer_tiling.md](/root/capstone-yonsei/flashinfer_tiling.md:8)를 참고하면 된다. 이 문서는 그보다 한 단계 위에서 "FlashInfer가 serving 시스템 안에서 정확히 무엇을 맡는가"를 설명하는 개요 문서다.

---

## 10. plan() 내부의 타일링 설정 로직

`self._cached_module.plan()` 호출은 아래 체인으로 이어진다.

```
Python plan() → C++ BatchPrefillWithKVCachePlan (TVM-FFI) → PrefillPlan<IdType>()
```

핵심 스케줄링 로직은 모두 `include/flashinfer/attention/scheduler.cuh`에 있다.

### 관련 코드 위치

| 역할 | 파일 |
|------|------|
| Python plan() 구현 | `flashinfer/prefill.py:1662~` |
| C++ TVM-FFI 바인딩 | `csrc/batch_prefill_jit_binding.cu` |
| C++ plan 구현체 | `csrc/batch_prefill.cu` |
| 핵심 스케줄링 로직 | `include/flashinfer/attention/scheduler.cuh` |
| CTA Tile Q 결정 함수 | `include/flashinfer/utils.cuh:384~403` |
| SM90 런타임 Tile Scheduler | `include/flashinfer/attention/hopper/tile_scheduler.cuh` |

### FA2 백엔드 타일링 흐름

**Step 0: GPU 리소스 조회** (`scheduler.cuh:712~720`)

```cpp
cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id);
int num_blocks_per_sm = 2;
int64_t available_ctas = num_blocks_per_sm * num_sm - num_colocated_ctas;
uint32_t max_batch_size_if_split = max_grid_size / num_kv_heads;
```

SM 개수를 쿼리해서 최대 동시 실행 가능한 CTA 수(`max_batch_size_if_split`)를 계산한다. 이것이 이후 타일 분할의 상한선이 된다.

**Step 1: `cta_tile_q` 결정** (`utils.cuh:384~403`, `FA2DetermineCtaTileQ`)

```
avg_packed_qo_len > 64 AND head_dim < 256  →  cta_tile_q = 128
else:
  compute_capability >= 8 (Ampere 이상):
    avg_packed_qo_len > 16  →  cta_tile_q = 64
    avg_packed_qo_len <= 16 →  cta_tile_q = 16
  compute_capability < 8 (Turing):
    cta_tile_q = 64  (SMEM 제약)
```

- 일반 모드: `sum(qo_len * gqa_group_size) / batch_size`로 평균 계산
- CUDA Graph 모드: `max_seq_len = total_rows - batch_size + 1` (최악의 경우 추정)

**Step 2: `kv_chunk_size` 결정 (Split-KV)** (`scheduler.cuh:101~130`, `PrefillBinarySearchKVChunkSize`)

이진 탐색으로 아래 조건을 만족하는 최소 `kv_chunk_size`를 찾는다.

```
new_batch_size = Σ ceil_div(packed_qo_len[i], cta_tile_q) * ceil_div(kv_len[i], kv_chunk_size)
new_batch_size <= max_batch_size_if_split
```

`kv_chunk_size`가 클수록 split 수가 줄어 `new_batch_size`가 감소한다. GPU CTA 수를 초과하지 않는 범위에서 최소값을 찾는 것이 목표다.

**Step 3: Work Partition 생성** (`scheduler.cuh:576~600`)

```cpp
for each request_idx:
    for q_tile_idx in range(ceil_div(packed_qo_len, cta_tile_q)):
        for kv_tile_idx in range(ceil_div(kv_len, kv_chunk_size)):
            request_indices.push_back(request_idx)
            qo_tile_indices.push_back(q_tile_idx)
            kv_tile_indices.push_back(kv_tile_idx)
```

각 `(request, q_tile, kv_tile)` 조합이 GPU의 하나의 CTA 작업으로 변환된다.

### `PrefillPlanInfo` 구조체 (`scheduler.cuh:615~691`)

plan()의 최종 결과물로, 이후 run()에서 그대로 커널에 전달된다.

```cpp
struct PrefillPlanInfo {
    int64_t cta_tile_q;              // Q 타일 크기
    int64_t padded_batch_size;       // Split 후 확장된 배치 크기
    int64_t request_indices_offset;  // 각 CTA가 처리할 request
    int64_t qo_tile_indices_offset;  // 각 CTA의 Q 타일 인덱스
    int64_t kv_tile_indices_offset;  // 각 CTA의 KV 타일 인덱스
    int64_t merge_indptr_offset;     // Split-KV 결과 병합 위치
    int64_t kv_chunk_size_ptr_offset;
    bool split_kv;
    // ...
};
```

### FA2 vs SM90 (Hopper) 타일링 차이

| 항목 | FA2 | SM90 (FA3/Hopper) |
|------|-----|-------------------|
| cta_tile_q | 동적 결정 (16 / 64 / 128) | 고정값 (128, head_dim=64이면 192) |
| 스케줄링 방식 | CPU에서 단순 균등 분할 | Min-Heap 로드 밸런싱 |
| KV Split | Binary Search로 결정 | 없음 |
| Cost Function | 없음 | `2 * qo_len + kv_len` |
| Causal 처리 | 타일 할당 전 고려 | `effective_kv_len`으로 cost 계산에 통합 |

SM90에서는 `PrefillSM90Plan()` (`scheduler.cuh:870~1019`)이 Min-Heap으로 각 SM에 작업을 할당한다. KV 길이 내림차순으로 정렬한 뒤 가장 비용이 낮은 SM에 순서대로 배분하는 방식이다.

---

## 11. FlashInfer 수정 계획 — CTA_TILE_Q 실험

### 배경

FA2 백엔드는 현재 CTA_TILE_Q로 **16, 64, 128** 세 값만 지원한다. 이 값은 `FA2DetermineCtaTileQ()`가 배치 통계에 기반해 런타임에 선택하고, `DISPATCH_CTA_TILE_Q` 매크로가 switch-case로 커널 템플릿을 인스턴스화한다. 새 타일 크기를 실험하려면 이 구조 전체를 이해하고 일관되게 수정해야 한다.

---

### 핵심 불변 조건 (invariant)

새 타일 크기를 추가할 때 반드시 지켜야 할 조건:

```
CTA_TILE_Q = NUM_WARPS_Q × NUM_MMA_Q × 16
```

**왜 이 조건이 존재하는가:**

Tensor Core MMA 명령(`mma.sync.aligned.m16n8k16`)은 M 방향(Q 방향)으로 **반드시 16행**을 처리한다. 이것은 하드웨어에 고정된 명령어 형식이다. 커널 내부에서 각 warp는 `NUM_MMA_Q`번의 MMA를 수행하고, `NUM_WARPS_Q`개의 warp가 CTA 내에서 Q 방향을 분담한다. 따라서 CTA가 한 번에 처리하는 Q 행 수는 반드시 이 세 값의 곱이어야 한다. 이 조건을 어기면 각 warp가 담당해야 할 행 수가 MMA 명령의 최소 단위인 16의 배수가 되지 않아, 틀린 메모리 주소를 읽거나 silently wrong result가 발생한다.

추가로 총 warp 수는 고정되어 있다:

```
NUM_WARPS_Q × NUM_WARPS_KV = 4  (총 128 threads/block)
```

따라서 가능한 CTA_TILE_Q 값은 오직 **16의 배수**이며, 현실적으로 아래 표와 같다.

| CTA_TILE_Q | NUM_WARPS_Q | NUM_WARPS_KV | NUM_MMA_Q | 불변 조건 검산 |
|---|---|---|---|---|
| 16 | 1 | 4 | 1 | 1×1×16 = 16 ✓ |
| 32 | 2 | 2 | 1 | 2×1×16 = 32 ✓ |
| 64 | 4 | 1 | 1 | 4×1×16 = 64 ✓ |
| 128 | 4 | 1 | 2 | 4×2×16 = 128 ✓ |
| 256 | 4 | 1 | 4 | 4×4×16 = 256 — 레지스터 한계로 실질 불가 |

**4, 8은 16의 배수가 아니므로 불가능하다.** 256은 구조적으로는 성립하지만 아래 레지스터 제약으로 막힌다.

---

### 레지스터 제약 (IsInvalid 조건)

`prefill.cuh:137`의 `KernelTraits::IsInvalid()`:

```cpp
NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256
→ 이 조건이 true이면 해당 커널 변형은 컴파일에서 제외된다
```

`NUM_MMA_D_VO = head_dim / 16`이고 `DTypeQKAccum`은 보통 float32(4바이트)이다. head_dim=128 기준으로 계산하면:

| CTA_TILE_Q | NUM_MMA_Q | NUM_MMA_KV=1일 때 값 | Invalid? |
|---|---|---|---|
| 64 | 1 | 1×(64+8) = 72 | ✓ 유효 |
| 128 | 2 | 2×(64+8) = 144 | ✓ 유효 |
| 256 | 4 | 4×(64+8) = 288 ≥ 256 | ✗ 항상 무효 |

CTA_TILE_Q=256은 head_dim=128에서 NUM_MMA_KV가 어떤 값이어도 레지스터 한계를 넘는다. 이 제약은 GPU warp당 레지스터 파일 크기를 반영한 물리적 한계다.

---

### 수정이 필요한 파일과 이유

**1. `include/flashinfer/utils.cuh:113` — DISPATCH_CTA_TILE_Q 매크로**

```cpp
// 추가 필요
case 32: {
  constexpr uint32_t CTA_TILE_Q = 32;
  __VA_ARGS__
  break;
}
```

이유: 런타임 값 32를 `constexpr` 템플릿 파라미터로 변환하는 분기가 없으면 `default` 분기에서 즉시 런타임 에러가 발생한다.

**2. `include/flashinfer/attention/prefill.cuh:55` — `get_num_warps_q`**

```cpp
// 현재
constexpr uint32_t get_num_warps_q(const uint32_t cta_tile_q) {
  if (cta_tile_q > 16) return 4;
  else return 1;
}

// 수정
constexpr uint32_t get_num_warps_q(const uint32_t cta_tile_q) {
  if (cta_tile_q > 32) return 4;
  else if (cta_tile_q > 16) return 2;  // CTA_TILE_Q=32: 2×1×16=32 ✓
  else return 1;
}
```

이유: 이 함수를 수정하지 않으면 CTA_TILE_Q=32일 때 NUM_WARPS_Q=4가 되어 각 warp가 8행만 담당하게 된다. 그러나 MMA 명령은 16행 단위이므로 불변 조건 `4×1×16=64≠32`를 위반한다. `get_num_warps_kv`는 `4 / get_num_warps_q`로 자동 계산되므로 별도 수정이 불필요하다.

**3. `include/flashinfer/utils.cuh:384` — `FA2DetermineCtaTileQ`**

```cpp
// 32를 반환하는 조건 삽입 (예시)
if (avg_packed_qo_len > 16 && avg_packed_qo_len <= 32) return 32;
```

이유: 이 함수가 32를 반환하지 않으면 실제 실행 경로가 32 커널을 타지 않는다. 실험 목적으로는 특정 조건을 통째로 `return 32`로 강제해도 된다.

**4. Jinja 템플릿 — `csrc/batch_prefill_paged_kernel_inst.jinja` 등**

```jinja
{# 기존 #}
{% for cta_tile_q in [16, 64, 128] %}

{# 수정 #}
{% for cta_tile_q in [16, 32, 64, 128] %}
```

이유: Jinja 템플릿이 커널의 explicit instantiation을 생성한다. 여기에 32가 없으면 DISPATCH_CTA_TILE_Q가 `case 32:`를 만나도 링크 에러가 발생한다.

---

### CTA_TILE_KV가 결정되는 흐름

CTA_TILE_Q와 달리 CTA_TILE_KV는 직접 선택하는 값이 아니다. 커널 내부에서 SMEM과 레지스터 제약 안에서 **자동으로 최대화**된다.

```
CTA_TILE_KV = NUM_MMA_KV × NUM_WARPS_KV × 16
```

**NUM_WARPS_KV**는 CTA_TILE_Q에 의해 이미 결정된다 (`4 / NUM_WARPS_Q`).

**NUM_MMA_KV**는 두 제약의 최솟값으로 결정된다 (`prefill.cuh:1634~1644`):

```
① SMEM 제약
  Q가 차지한 SMEM을 제외한 나머지 예산으로 KV tile을 몇 개 올릴 수 있는지:
  max_num_mma_kv_smem = (max_smem_per_block - CTA_TILE_Q × HEAD_DIM_QK × sizeof(DTypeQ))
                        / ((HEAD_DIM_QK + HEAD_DIM_VO) × 16 × NUM_WARPS_KV × sizeof(DTypeKV))

② 레지스터 제약
  max_num_mma_kv_reg = 8 / NUM_MMA_Q
  (단, head_dim≥128 + NUM_MMA_Q=2 + RoPE 조합이면 2로 하향)

NUM_MMA_KV = DISPATCH_NUM_MMA_KV( min(①, ②) )  →  {8, 4, 2, 1} 중 선택
```

예시 (CTA_TILE_Q=128, head_dim=128, fp16, A100):
- NUM_WARPS_KV=1, NUM_MMA_Q=2
- max_num_mma_kv_reg = 8/2 = 4
- SMEM 예산 ≈ 50KB, KV tile 하나 = 8KB → max_num_mma_kv_smem ≈ 6
- NUM_MMA_KV = min(6, 4) = 4 → **CTA_TILE_KV = 4×1×16 = 64**

CTA_TILE_KV를 실험적으로 줄이려면 `DISPATCH_NUM_MMA_KV` 호출부에서 값을 강제 지정하면 된다:
```cpp
// 현재: 제약 안에서 자동 최대화
DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, { ... })
// 실험: 강제 지정
DISPATCH_NUM_MMA_KV(2, NUM_MMA_KV, { ... })
```

---

### 수정 후 동작을 예상하는 근거

- `CTA_TILE_Q=32, NUM_WARPS_Q=2, NUM_MMA_Q=1`은 불변 조건 `2×1×16=32`를 만족한다
- `IsInvalid()` 계산: `1×(8×8 + 2×4×NUM_MMA_KV)`는 head_dim=128에서 NUM_MMA_KV≤23이면 통과한다
- SMEM 사용량: `32×128×2 = 8KB`로 기존 64KB(CTA_TILE_Q=128) 대비 훨씬 작아 제약이 없다
- 스케줄러(`scheduler.cuh`)는 cta_tile_q를 숫자로만 다루므로 별도 수정 없이 32를 그대로 처리한다

---

### 수정 후 실행 절차

```bash
rm -rf ~/.cache/flashinfer/   # JIT 캐시 삭제 (반드시 필요)
python -c "import flashinfer"  # 재컴파일 트리거
```

검증 방법: 기존 결과와 수치 비교로 정확도 확인, `bench_attention.py`로 throughput 비교.
